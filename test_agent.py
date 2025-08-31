"""
Integrated testable tools + agent for the Travel Planner project.

Run this as a standalone script to test each tool individually and then run
an interactive agent loop.

Requirements (pip):
  pip install amadeus googlemaps serpapi requests python-dotenv langchain langchain-ollama pydantic

Notes:
- The script reads keys from environment (.env recommended).
- Tools implement small, well-documented outputs and return compact summaries
  suitable for feeding into an LLM.
- The agent uses ChatOllama as the LLM (you used this earlier). Swap to another
  LLM class if you prefer.

"""

import os
import time
import requests
from typing import List, Dict, Any, Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic import PrivateAttr

# LangChain / Ollama imports
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
# 3rd party APIs
from amadeus import Client as AmadeusClient
import googlemaps
from serpapi import GoogleSearch

load_dotenv()

# --- Configuration / Credentials ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# --- Utility helpers ---

def iso_to_local(dt_str: str) -> str:
    """Return the timestamp as-is for now (placeholder for timezone formatting)."""
    return dt_str


def format_duration(pt_duration: str) -> str:
    """Convert ISO-8601 duration like 'PT12H30M' into human readable."""
    # crude parser
    s = pt_duration.replace("PT", "")
    hours = "0"
    mins = "0"
    if "H" in s:
        parts = s.split("H")
        hours = parts[0]
        s = parts[1]
    if "M" in s:
        mins = s.replace("M", "")
    return f"{hours}h {mins}m"


# --- Tool: Currency converter ---
class CurrencyArgs(BaseModel):
    amount: float = Field(..., description="Amount to convert")
    from_currency: str = Field(..., description="Currency code, e.g. 'USD'")


class CurrencyTool(BaseTool):
    name: str = "convert_to_inr"
    description: str = (
        "While searching for Hotels and flights, convert fares in any currency "
        "to INR (Indian Rupees) using currencyapi.com. "
        "Do NOT call this if the price is already in INR."
    )
    args_schema: Type[BaseModel] = CurrencyArgs

    def _run(self, amount: float, from_currency: str) -> Dict[str, Any]:
        # 🔹 Prevent unnecessary conversion if already INR
        if from_currency.upper() == "INR":
            return {
                "amount": amount,
                "from_currency": "INR",
                "rate": 1.0,
                "inr": round(amount, 2),
                "note": "Already in INR — no conversion performed."
            }

        if not CURRENCY_API_KEY:
            raise RuntimeError("Missing CURRENCY_API_KEY in environment")

        params = {
            "apikey": CURRENCY_API_KEY,
            "base_currency": from_currency,
            "currencies": "INR"
        }
        r = requests.get("https://api.currencyapi.com/v3/latest", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        rate = data["data"]["INR"]["value"]
        converted = round(amount * rate, 2)
        return {
            "amount": amount,
            "from_currency": from_currency,
            "rate": rate,
            "inr": converted
        }

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError



# --- Tool: Google Maps sightseeing ---
class SightArgs(BaseModel):
    city: str = Field(..., description="City name, e.g. 'Paris'")
    max_results: int = Field(5, description="Max number of results")


class SightseeingTool(BaseTool):
    name: str = "fetch_sightseeing"
    description: str = "Return top tourist attractions (name, address, rating) for a city using Google Maps Places API."
    args_schema: Type[BaseModel] = SightArgs

    def __init__(self):
        super().__init__()
        if not GOOGLE_MAPS_API_KEY:
            self._client = None
        else:
            self._client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

    def _run(self, city: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self._client:
            raise RuntimeError("Missing GOOGLE_MAPS_API_KEY")
        # lightweight call and summary
        resp = self._client.places(query=f"tourist attractions in {city}")
        results = resp.get("results", [])[:max_results]
        summary = []
        for p in results:
            summary.append({
                "name": p.get("name"),
                "address": p.get("formatted_address"),
                "rating": p.get("rating")
            })
        return summary

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


# --- Tool: OpenWeather daily min/max ---
class WeatherArgs(BaseModel):
    city: str = Field(..., description="City name to fetch weather for")


class WeatherTool(BaseTool):
    name: str = "fetch_weather"
    description: str = "Return min/max temperatures per day (Celsius) for the next days using OpenWeather forecast endpoint."
    args_schema: Type[BaseModel] = WeatherArgs

    def _run(self, city: str) -> Dict[str, Dict[str, float]]:
        if not OPENWEATHER_API_KEY:
            raise RuntimeError("Missing OPENWEATHER_API_KEY")
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        r = requests.get("https://api.openweathermap.org/data/2.5/forecast", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        daily = {}
        for entry in data.get("list", []):
            date = entry["dt_txt"].split(" ")[0]
            temps = daily.setdefault(date, [])
            temps.append(entry["main"]["temp"])
        return {d: {"min": min(t), "max": max(t)} for d, t in daily.items()}

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


# --- Tool: SerpAPI search (generic) ---
class SerpArgs(BaseModel):
    q: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Max results to return")


class SerpTool(BaseTool):
    name: str = "serpapi_search"
    description: str = "Run a Google search via SerpAPI and return top links."
    args_schema: Type[BaseModel] = SerpArgs

    def _run(self, q: str, max_results: int = 5) -> List[str]:
        if not SERPAPI_API_KEY:
            raise RuntimeError("Missing SERPAPI_API_KEY")
        params = {"q": q, "api_key": SERPAPI_API_KEY, "engine": "google"}
        search = GoogleSearch(params)
        data = search.get_dict()
        results = data.get("organic_results", [])
        return [r.get("link") for r in results[:max_results]]

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


# --- Tool: Amadeus flight search (with formatter) ---
class FlightArgs(BaseModel):
    origin: str = Field(..., description="Origin IATA code, e.g. 'BOM'")
    destination: str = Field(..., description="Destination IATA code, e.g. 'CDG'")
    departure_date: str = Field(..., description="Departure date YYYY-MM-DD")
    adults: int = Field(1, description="Number of adult travelers")


class AmadeusFlightTool(BaseTool):
    name: str = "amadeus_flight_search"
    description: str = "Search Amadeus for flights to Destination  offers and return a compact human-readable summary list."
    args_schema: Type[BaseModel] = FlightArgs

    _amadeus: AmadeusClient = PrivateAttr()

    def __init__(self):
        super().__init__()
        if not (AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET):
            self._amadeus = None
        else:
            self._amadeus = AmadeusClient(
                client_id=AMADEUS_CLIENT_ID,
                client_secret=AMADEUS_CLIENT_SECRET
            )

    def _run(self, origin: str, destination: str, departure_date: str, adults: int = 1) -> List[Dict[str, Any]]:
        if not self._amadeus:
            raise RuntimeError("Missing Amadeus credentials in environment")
        
        # call
        resp = self._amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            adults=adults
        )
        raw = resp.data
        
        # format
        pretty = []
        for offer in raw:
            price = offer.get("price", {}).get("grandTotal") or offer.get("price", {}).get("total")
            currency = offer.get("price", {}).get("currency")
            itineraries = []
            for itin in offer.get("itineraries", []):
                it_segs = []
                for seg in itin.get("segments", []):
                    it_segs.append({
                        "from": seg["departure"]["iataCode"],
                        "to": seg["arrival"]["iataCode"],
                        "dep": iso_to_local(seg["departure"]["at"]),
                        "arr": iso_to_local(seg["arrival"]["at"]),
                        "carrier": seg.get("carrierCode"),
                        "flight_no": seg.get("number"),
                        "duration": format_duration(seg.get("duration", "PT0M"))
                    })
                itineraries.append({
                    "duration": format_duration(itin.get("duration", "PT0M")),
                    "segments": it_segs
                })
            pretty.append({"price": f"{price} {currency}", "itineraries": itineraries})
        
        return pretty

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


# --- Agent initialization (tool-calling agent) ---


llm = ChatOllama(model=OLLAMA_MODEL, format="json")


    # Define a small system instruction describing tool usage
system_prompt = """
You are a travel planning assistant. Specifically for Indian Users, 
Use available tools to search for flights to the destination specified by the user in their query 
fetch flight offers, hotels and sightseeing,
weather forecasts, and currency conversions. Always summarize API results in a human-friendly way
before presenting them to the user. Use tools only when necessary. Keep user budget and preferences in memory.
"""
from langchain import LLMChain, PromptTemplate
# Wrap the system prompt in a PromptTemplate
template = PromptTemplate(
    input_variables=["input", "history"],
    template=f"""{system_prompt}

Conversation history:
{{history}}

User: {{input}}
Assistant:"""
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [
        SerpTool(),
        CurrencyTool(),
        SightseeingTool(),
        WeatherTool(),
        AmadeusFlightTool()]
    
    # Create agent chain with the system prompt included
agent_chain = initialize_agent(tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory)

# --- If run as script: test tools and run interactive loop ---
if __name__ == "__main__":
    # agent = build_agent()
    print("Indian Travel Planner Assistant (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        response = agent_chain.run(user_input)
        print("\nAssistant:", response)
