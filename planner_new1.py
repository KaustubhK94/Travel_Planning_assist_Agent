### Planner_new1.py
import os
import json
import re
import time
from typing import Optional, Type, List, Dict, Any
from datetime import datetime, timedelta
from uu import Error

from langchain_core.messages import trim_messages
import requests
import googlemaps
from amadeus import Client
from dotenv import load_dotenv

# LangChain (core + community + ollama)
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from pydantic import BaseModel, Field
from serpapi import GoogleSearch
from langchain_community.llms import Ollama
# =============================
# Load configuration / clients
# =============================
load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
GOOGLE_MAPS_API_KEY   = os.getenv("GOOGLE_MAPS_API_KEY")

# Fail-fast if critical keys are missing
missing = [k for k, v in {
    "SERPAPI_API_KEY": SERPAPI_API_KEY,
    "CURRENCY_API_KEY": CURRENCY_API_KEY,
    "GOOGLE_MAPS_API_KEY": GOOGLE_MAPS_API_KEY,
    "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
    "AMADEUS_CLIENT_ID": AMADEUS_CLIENT_ID,
    "AMADEUS_CLIENT_SECRET": AMADEUS_CLIENT_SECRET,
}.items() if not v]
if missing:
    print("[WARN] Missing keys:", ", ".join(missing))

# External SDK clients
amadeus = Client(client_id=AMADEUS_CLIENT_ID, client_secret=AMADEUS_CLIENT_SECRET)

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


# =============================
# Trip Memory Class (moved before ExtractTripDetailsTool)
# =============================

class TripMemory(BaseModel):
    origin: str | None = None
    destination: str | None = None
    destination_city: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    adults: int = 1
    preferences: List[str] = Field(default_factory=list)

    def update(self, details: dict):
        for k, v in details.items():
            if v is not None and v != "":
                setattr(self, k, v)

# =============================
# Utilities
# =============================

def _extract_json_block(text: str) -> str:
    """Return the first top-level JSON object substring found in text.
    Gracefully handles extra prose by finding the outermost {...} block.
    """
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found in LLM output")
    return text[first:last+1]


def _format_duration(iso_duration: str) -> str:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", iso_duration or "")
    if not m:
        return iso_duration
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    return f"{h}h {m_}m"


def _fmt_time(iso_time: str) -> str:
    try:
        return datetime.fromisoformat(iso_time.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_time


# =============================
# Tool Schemas
# =============================

class FlightSearchInput(BaseModel):
    origin: str = Field(..., description="IATA code of departure airport (e.g., DEL)")
    destination: str = Field(..., description="IATA code of arrival airport (e.g., CDG)")
    start_date: str = Field(..., description="Departure date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="Return date in YYYY-MM-DD format (optional)")
    currency: str = Field(default="USD", description="Currency code for pricing (e.g., USD, INR)")


class HotelSearchInput(BaseModel):
    destination_city: str = Field(..., description="City name of the destination (e.g., Paris)")
    check_in_date: str = Field(..., description="Hotel check-in date in YYYY-MM-DD format")
    check_out_date: str = Field(..., description="Hotel check-out date in YYYY-MM-DD format")
    adults: int = Field(default=2, description="Number of adults")


class CurrencyConversionInput(BaseModel):
    amount: float = Field(..., description="The amount of money to convert.")
    from_currency: str = Field(..., description="The currency code to convert from (e.g., USD)")


class SightseeingInput(BaseModel):
    city: str = Field(..., description="The city to get sightseeing attractions for")


class WeatherInput(BaseModel):
    city: str = Field(..., description="City name for weather forecast.")
    start_date: str = Field(..., description="Trip start date in YYYY-MM-DD format.")
    end_date: str = Field(..., description="Trip end date in YYYY-MM-DD format.")


class ExtractTripDetailsInput(BaseModel):
    user_query: str = Field(..., description="Free-form user request about a trip")

# =============================
# Tools (BaseTool subclasses)
# =============================

class SearchFlightsTool(BaseTool):
    name: str = "search_flights"
    description: str = (
        "Search for flights using the Amadeus API. Useful when you have origin, destination, "
        "and dates. Returns up to 5 best options with prices and segments."
    )
    args_schema: Type[BaseModel] = FlightSearchInput

    def _run(
        self,
        origin: str,
        destination: str,
        start_date: str,
        end_date: Optional[str] = None,
        currency: str = "USD",
    ) -> str:
        try:
            params = {
                "originLocationCode": origin,
                "destinationLocationCode": destination,
                "departureDate": start_date,
                "adults": 1,
                "currencyCode": currency,
                "max": 5,
            }
            if end_date:
                params["returnDate"] = end_date

            response = amadeus.shopping.flight_offers_search.get(**params)
            data = response.data

            def _format_duration(iso):
                if not iso:
                    return "?"
                match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", iso)
                if not match:
                    return iso
                h = int(match.group(1) or 0)
                m = int(match.group(2) or 0)
                return f"{h}h {m}m"

            def _fmt_time(iso):
                try:
                    return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
                except:
                    return iso

            def _legs(segments):
                legs = []
                for seg in segments:
                    dep = seg["departure"]["iataCode"]
                    arr = seg["arrival"]["iataCode"]
                    dep_t = _fmt_time(seg["departure"]["at"])
                    arr_t = _fmt_time(seg["arrival"]["at"])
                    dur = _format_duration(seg.get("duration"))
                    legs.append(f"{dep} ({dep_t}) → {arr} ({arr_t}) [{dur}]")
                return legs

            if not data:
                return "No flights found."

            lines = [f"Found {len(data)} flight options:\n"]
            for offer in data:
                price = f"{offer['price']['total']} {offer['price']['currency']}"
                airline = (offer.get("validatingAirlineCodes") or ["?"])[0]
                itins = offer.get("itineraries", [])
                outbound = _legs(itins[0]["segments"]) if len(itins) > 0 else []
                ret = _legs(itins[1]["segments"]) if len(itins) > 1 else []

                lines.append(f"💰 {price} | ✈ {airline}")
                lines.append("  Outbound:")
                for leg in outbound:
                    lines.append(f"    {leg}")
                if ret:
                    lines.append("  Return:")
                    for leg in ret:
                        lines.append(f"    {leg}")
                lines.append("")  # blank line between offers

            return "\n".join(lines)

        except Exception as e:
            return f"Error searching flights: {e}"

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not implemented for SearchFlightsTool")


class SearchHotelsTool(BaseTool):
    name: str = "search_hotels"
    description: str = "Find hotels in a given destination city via SerpAPI Google Hotels."
    args_schema: Type[BaseModel] = HotelSearchInput

    def _run(self, destination_city: str, check_in_date: str, check_out_date: str, adults: int = 2) -> List[Dict[str, Any]]:
        params = {
            "engine": "google_hotels",
            "q": destination_city,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "currency": "USD",
            "gl": "us",
            "hl": "en",
            "api_key": SERPAPI_API_KEY,
        }
        try:
            resp = requests.get("https://serpapi.com/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return [{"error": f"SerpAPI error: {e}"}]

        hotels = []
        for h in (data or {}).get("properties", [])[:15]:
            hotels.append({
                "name": h.get("name"),
                "price": (h.get("rate_per_night") or {}).get("lowest"),
                "rating": h.get("overall_rating"),
                "address": h.get("address"),
                "link": h.get("link"),
            })
        return hotels

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


class CurrencyConvertTool(BaseTool):
    name: str = "convert_currency"
    description: str = "Convert currency amounts"
    args_schema: Type[BaseModel] = CurrencyConversionInput

    def _run(self, amount: float, from_currency: str, to_currency: str = "INR"):
        if from_currency.upper() == to_currency.upper():
            return {"amount": amount, "currency": to_currency.upper()}

        params = {
            "apikey": CURRENCY_API_KEY,
            "base_currency": from_currency,
            "currencies": to_currency
        }
        data = requests.get("https://api.currencyapi.com/v3/latest", params=params).json()
        rate = data["data"][to_currency.upper()]["value"]
        converted = round(amount * rate, 2)

        return {
            "original": f"{amount} {from_currency.upper()}",
            "converted": f"{converted} {to_currency.upper()}",
            "rate": rate
        }

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


class FetchSightseeingTool(BaseTool):
    name: str = "fetch_sightseeing"
    description: str = "Get top tourist attractions for a given city using SerpAPI Google Maps."
    args_schema: Type[BaseModel] = SightseeingInput

    def _run(self, city: str) -> str:
        print(f"[DEBUG] FetchSightseeingTool executed with city={city}")
        try:
            params = {
                "engine": "google_maps",
                "q": f"tourist attractions in {city}",
                "type": "search",
                "api_key": os.getenv("SERPAPI_API_KEY"),
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            # ✅ check both local_results and places_results
            places = results.get("local_results", []) or results.get("places_results", [])

            if not places:
                return f"No sightseeing places found for {city}."

            attractions = []
            for p in places[:10]:
                attractions.append(
                    f"- {p.get('title', 'Unknown')} "
                    f"(Rating: {p.get('rating', 'N/A')}, "
                    f"Address: {p.get('address', 'N/A')})"
                )

            return f"Top sightseeing attractions in {city}:\n" + "\n".join(attractions)

        except Exception as e:
            return f"[ERROR] FetchSightseeingTool failed: {str(e)}"


class FetchWeatherTool(BaseTool):
    name: str = "fetch_weather"
    description: str = "Get detailed weather forecast for a city & date range using OpenWeather 5-day/3-hour API."
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str, start_date: str, end_date: str) -> Dict[str, List[Dict[str, Any]]]:
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        try:
            data = requests.get("https://api.openweathermap.org/data/2.5/forecast", params=params).json()
        except Exception as e:
            return {"error": str(e)}  # type: ignore

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        except Exception:
            return {"error": "Invalid date format. Use YYYY-MM-DD."}  # type: ignore

        filtered: Dict[str, List[Dict[str, Any]]] = {}
        for entry in (data or {}).get("list", []):
            try:
                entry_dt = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            entry_date = entry_dt.date()
            if start_dt <= entry_date <= end_dt:
                day_str = entry_date.strftime("%Y-%m-%d")
                filtered.setdefault(day_str, []).append({
                    "time": entry_dt.strftime("%H:%M"),
                    "temp": round(entry["main"]["temp"], 1),
                    "feels_like": round(entry["main"]["feels_like"], 1),
                    "condition": (entry.get("weather") or [{"description": ""}])[0]["description"].title(),
                    "humidity": entry["main"].get("humidity"),
                    "wind_speed": (entry.get("wind") or {}).get("speed"),
                })
        return filtered

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


class ExtractTripDetailsTool(BaseTool):
    name: str = "extract_trip_details"
    description: str = (
        "Extract structured trip details from the user query. "
        "Always return origin and destination as IATA airport codes (3-letter airport codes). "
        "These can be used directly in flight search, without calling resolve_iata again."
    )
    trip_memory: "TripMemory"
    args_schema: Type[BaseModel] = ExtractTripDetailsInput


    def _run(self, user_query: str) -> Dict[str, Any]:
        prompt = f"""
You are a highly accurate travel assistant.
Return ONLY valid JSON (no prose). Extract these fields from the user's request and always fill them:
{{
  "origin": "<IATA code>",
  "destination": "<IATA code>",
  "destination_city": "<city name>",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "adults": 2
}}
Rules:
- Detect patterns like "from X to Y", "to Y from X", or city/country mentions.
- If a country is given, choose the busiest international airport.
- Convert to correct 3-letter IATA codes.
- Understand exact dates (e.g., "22 Aug 2025"), relative dates ("next Tuesday"), durations ("for 5 days").
- If year missing, assume next occurrence from today.
- If only start + length given, compute end_date.
- If travelers mentioned, set adults accordingly; else default to 2.
- Do NOT include any text outside JSON.

User: {user_query}
"""
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=90,
            )
            raw = (resp.json() or {}).get("response", "").strip()
            json_str = _extract_json_block(raw)
            details = json.loads(json_str)
            self.trip_memory.update(details)
            return details
        except Exception as e:
            # Fallback minimal structure to keep agent moving
            return {
                "origin": None,
                "destination": None,
                "destination_city": None,
                "start_date": None,
                "end_date": None,
                "adults": 2,
                "_error": str(e),
            }

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError



from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(
            f'<div class="msg-container assistant">'
            f'<div class="assistant-msg">{self.text}</div></div>',
            unsafe_allow_html=True,
        )


# =============================
# Build the ReAct Agent
# =============================

def build_agent(verbose: bool = True):

#     # ---- LLM with system prompt ----
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        num_ctx=5024,
        # reasoning=True,
        temperature=0.2,
        callbacks=[StreamingStdOutCallbackHandler()],
        verbose=verbose,   
    )

#     # ---- Domain memory ----
    trip_memory = TripMemory()

    # ---- Tools ----
    tools = [
        SearchFlightsTool(),
        SearchHotelsTool(),
        FetchWeatherTool(),
        FetchSightseeingTool(),
        CurrencyConvertTool(),
        ExtractTripDetailsTool(trip_memory=trip_memory),
    ]

#     # ---- Conversation memory ----
    memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    # ai_prefix="Agent",
    # human_prefix="User",
     # max_token_limit=2000,
    output_key="output"
    )



#      ---- ReAct prefix ----
    prefix = """
    - You are TripMate, a Reasoning and Acting Agent.
        Chat history:
        {chat_history}
    Follow ReAct reasoning:
    - Understand User's intent first, Decide if it requires any tool/tools call, If yes first use extract trip details and then decide the best tool/tools to call,
    - See if you could answer the Follow Up question Based on the Memory provided to you. 
    - Always give the final answer in plain natural language, never return JSON to the user.
    - For tool calls, use structured JSON. For final answers, only output natural language text.
    1. Understand the user’s intent is it full trip planning, or just searching for flights ,hotels, weather, sightseeing or tourist attractions.
    2. Decide the best tool or tools to call based on user request.
    3. Call tool → observe result.
    4. Repeat if needed until you can provide a complete answer.
    5. Never make up answers if a tool exists that can provide the info.
    6. Your Aim is to Guide user from the information you recieved after calling the tool/tools.
    7. answer the follow up question, reuse relevant details from chat_history.
    Rules:
    - If the request is for a single task (e.g. only hotels or only weather), call only that tool.
    - If the request is similar to “plan a trip”, first call extract_trip_details, then flights, hotels, weather, sightseeing in that order.
    - When a tool returns structured data (like flight search, hotels, weather), always:
      * List multiple options with bullet points.
      * Include all important fields: 
        - Flights → airline, price, times, layovers, return trip.
        - Hotels → name, price per night, rating, location.
         - Weather → daily forecast with temperature, condition, humidity.
         - Sightseeing → name, rating, address.
       * Do NOT skip important details, even if multiple results exist. 
       * Explain the Outputs of the tools in a Detailed Meaningful manner. 
     - Never leave the final response blank.
     """

    # ---- Build ReAct Agent ----
    agent = initialize_agent(
         tools=tools,
         llm=llm,
         agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
         verbose=verbose,
         memory=memory,
         max_iterations=3,
         early_stopping_method="generate",
         handle_parsing_errors="There was an error parsing the model output. Please try again.",
         return_intermediate_steps=True,
         agent_kwargs={
             "prefix": prefix,
             "input_variables": ["input", "chat_history"]
         },
     )
    print("Registered tools:", [t.name for t in tools])
    return agent


# =============================
# CLI runner
# =============================

def main():
    agent = build_agent(verbose=True)
    print("\nTripMate ReAct Agent ready. Ask me to plan, search flights/hotels, check weather, or find sights.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user = input("you> ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit", ":q"}:
                print("Bye! Safe travels.")
                break

             # Run the agent
            out = agent.invoke({"input": user})
            print("\nagent>#", out["output"], "\n")

             # --- Extract final agent response safely ---
            final = out.get("output") or out.get("final_output") or out.get("result") or str(out)

             # --- Debug: show intermediate reasoning ---
            if isinstance(out, dict):
                steps = out.get("intermediate_steps", [])
                if steps:
                    for i, (action, obs) in enumerate(steps, 1):
                        print(f" Step {i}:")

                 # Explicit final answer line
                print("\nagent>#", final, "\n")

            else:
                 print("\nagent>>", final, "\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Bye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
