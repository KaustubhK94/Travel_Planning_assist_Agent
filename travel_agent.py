# =============================
# travel_agent.py
# =============================
import os
import json

import re
import time
from typing import Optional, Type, List, Dict, Any
import datetime
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
from dateutil import parser as dateparser

# =============================
# Load configuration / clients
# =============================
load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

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

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
#llama3.1:8b

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


def normalize_date(date_str: str) -> str:
    dt = dateparser.parse(date_str, dayfirst=True)
    today = datetime.date.today()  # now really works

    if dt.year < today.year:
        dt = dt.replace(year=today.year)

    if dt.date() < today:
        dt = dt.replace(year=dt.year + 1)

    return dt.strftime("%Y-%m-%d")


def _extract_json_block(text: str) -> str:
    # Remove ```json or ``` fences
    text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE)

    # Find the first JSON-like block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in: {text[:200]}")
    return match.group(0)


class ExtractTripDetailsInput(BaseModel):
    user_query: str = Field(..., description="Natural language query from the user.")


class FlightSearchInput(BaseModel):
    user_query: str = Field(
        ...,
        description="The full natural language user query (e.g. 'Mumbai to Milan on 24 Aug returning 27 Aug')."
    )


class SearchFlightsTool(BaseTool):
    name: str = "search_flights"
    description: str = (
        "Search for flights. Input should be the full user query "
        "(e.g. 'find me flights from Delhi to Paris on 20 Aug returning on 25 Aug')."
    )
    args_schema: Type[BaseModel] = FlightSearchInput
    trip_memory: TripMemory

    def _run(self, user_query: str) -> str:
        # --- Step 1: Extract structured flight details using LLM ---
        prompt = f"""
        You're a JSON only extraction tool.
        Extract flight search details from the following query.
        Return ONLY valid JSON in this format:
        {{
          "origin": "<IATA code>",
          "destination": "<IATA code>",
          "start_date": "YYYY-MM-DD",
          "end_date": "YYYY-MM-DD or null",
          "currency": "USD"
        }}
        Rules:
        - Do not include markdown code fences (no ```json).
        - Do not include explanations or text outside the JSON.
        - Detect patterns like "from X to Y", "to Y from X", or city/country mentions.
        - Always Convert origin and destination to correct 3-letter IATA airport codes.
        - Dates must be exact ISO format (YYYY-MM-DD).
        - If a country is given, choose the busiest international airport.
        - Understand exact dates (e.g., "22 Aug 2025"), relative dates ("next Tuesday"), durations ("for 5 days") 
        - If only start + length given, compute end_date by adding the length to the start_date.
        - If no currency is mentioned, default to USD.
        - If the query says "for N days", compute end_date as start_date + N days.

        - If year is missing, assume the next occurrence from today.

        User: {user_query}
        """
        #,"format":"json"
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt ,"stream": False},
                timeout=60,
            )
            raw = (resp.json() or {}).get("response", "").strip()
            json_str = _extract_json_block(raw)
            details = json.loads(json_str)
            print("DEBUG: raw flight API response:",details)

            details["start_date"] = normalize_date(details["start_date"])
            if details.get("end_date") and details["end_date"] not in [None, "null"]:
                details["end_date"] = normalize_date(details["end_date"])


            # update trip memory (safe update)
            for k, v in details.items():
                if hasattr(self.trip_memory, k):
                    setattr(self.trip_memory, k, v)

        except Exception as e:
            return f"Failed to parse flight details: {e}"

        # --- Step 2: Call Amadeus API ---
        try:
            params = {
                "originLocationCode": details["origin"],
                "destinationLocationCode": details["destination"],
                "departureDate": details["start_date"],
                "adults": 1,
                "currencyCode": details.get("currency", "USD"),
                "max": 5,
            }
            if details.get("end_date") and details["end_date"] != "null":
                params["returnDate"] = details["end_date"]

            response = amadeus.shopping.flight_offers_search.get(**params)
            data = response.data

            if not data:
                return "No flights found."

            # Format results (same as your original)
            def _format_duration(iso):
                if not iso: return "?"
                match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", iso)
                if not match: return iso
                h = int(match.group(1) or 0)
                m = int(match.group(2) or 0)
                return f"{h}h {m}m"

            def _fmt_time(iso):
                try:
                    from datetime import datetime
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
                    flight_num = f"{seg['carrierCode']}{seg['number']}"
                    legs.append(f"{dep} ({dep_t}) → {arr} ({arr_t}) [{dur}] | Flight {flight_num}")
                return legs

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
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return f"Error searching flights: {e}"

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


class SearchHotelsTool(BaseTool):
    name: str = "search_hotels"
    description: str = (
        "Find hotels for a trip. "
        "Input should be a natural language query (e.g. "
        "'find hotels in Paris from 20th Aug to 25th Aug for 2 adults')."
    )
    args_schema: Type[BaseModel] = ExtractTripDetailsInput  # just user_query

    def _run(self, user_query: str) -> List[Dict[str, Any]]:
        # --- Step 1: Extract structured hotel details ---
        prompt = f"""
        You're a JSON only extraction tool.
        Extract hotel search details from the user query.
        Respond with ONLY valid JSON — no explanations, no extra text.
        {{
          "destination_city": "<city>",
          "check_in_date": "YYYY-MM-DD",
          "check_out_date": "YYYY-MM-DD",
          "adults": 2
        }}

        Rules:
        - Do not include markdown code fences (no ```json).
        - Do not include explanations or text outside the JSON.
        - City and dates must always be present.
        - Always include city and dates in ISO YYYY-MM-DD format.
        - Default adults = 2 if not specified.
        - If year missing, assume next occurrence.

        User: {user_query}
        """

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=60,
            )
            raw = (resp.json() or {}).get("response", "").strip()
            json_str = _extract_json_block(raw)
            details = json.loads(json_str)

            details["check_in_date"] = normalize_date(details["check_in_date"])
            details["check_out_date"] = normalize_date(details["check_out_date"])

            # Ensure checkout is after checkin
            check_in_dt = datetime.date.fromisoformat(details["check_in_date"])
            check_out_dt = datetime.date.fromisoformat(details["check_out_date"])
            if check_out_dt <= check_in_dt:
                check_out_dt = check_in_dt + datetime.timedelta(days=1)
                details["check_out_date"] = check_out_dt.strftime("%Y-%m-%d")


        except Exception as e:
            return [{"error": f"Failed to parse hotel details: {e}"}]

        # --- Step 2: Call SerpAPI Google Hotels ---
        params = {
            "engine": "google_hotels",
            "q": details["destination_city"],
            "check_in_date": details["check_in_date"],
            "check_out_date": details["check_out_date"],
            "adults": details.get("adults", 2),
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
        for h in (data or {}).get("properties", [])[:10]:
            hotels.append({
                "name": h.get("name"),
                "price": (h.get("rate_per_night") or {}).get("lowest"),
                "rating": h.get("overall_rating"),
                "address": h.get("address")
            })

        return hotels

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


class CurrencyConvertTool(BaseTool):
    name: str = "convert_currency"
    description: str = (
        """Help User with currency exchange related queries,
        or help user manage their trips according to their 
        budgets. Inputs will be user query. 
        """
    )
    args_schema: Type[BaseModel] = ExtractTripDetailsInput  # just user_query

    def _run(self, user_query: str):
        # --- Step 1: Extract structured currency conversion request ---
        prompt = f"""
    You are a JSON-only extraction tool.
    Extract currency conversion details from the query.

    Return ONLY a valid JSON object in this format:
    {{
    "amount": 100.0,
    "from_currency": "USD",
    "to_currency": "INR"
    }}

    Rules:
    - Do not include markdown code fences (no ```json).
    - Do not include explanations or text outside the JSON.
    - Do not include markdown code fences.
    - Do not include explanations or text outside JSON.
    - If no amount is given, use 1.
    - If no target currency is given, default to INR.

    Query: "{user_query}"
    """


        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=60,
            )
            raw = (resp.json() or {}).get("response", "").strip()
            json_str = _extract_json_block(raw)
            details = json.loads(json_str)


        except Exception as e:
            return {"error": f"Failed to parse currency details: {e}"}

        # --- Step 2: Currency conversion API call ---
        from_currency = details["from_currency"].upper()
        to_currency = details.get("to_currency", "INR").upper()
        amount = float(details["amount"])

        if from_currency == to_currency:
            return {"amount": amount, "currency": to_currency}

        params = {
            "apikey": CURRENCY_API_KEY,
            "base_currency": from_currency,
            "currencies": to_currency
        }

        try:
            data = requests.get("https://api.currencyapi.com/v3/latest", params=params).json()
            rate = data["data"][to_currency]["value"]
            converted = round(amount * rate, 2)
        except Exception as e:
            return {"error": f"Currency API error: {e}"}

        return {
            "original": f"{amount} {from_currency}",
            "converted": f"{converted} {to_currency}",
            "rate": rate
        }

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


class FetchSightseeingTool(BaseTool):
    name: str = "fetch_sightseeing"
    description: str = "Get top tourist attractions for a given city using SerpAPI Google Maps."
    args_schema: Type[BaseModel] = ExtractTripDetailsInput  # single query input

    def _run(self, user_query: str) -> str:
        print(f"[DEBUG] FetchSightseeingTool executed with query={user_query}")
        try:
            # --- Extract city from user query ---
            prompt = f"Extract only the city name from this query: '{user_query}'. Respond with only the city."
            city = ChatOllama(model=OLLAMA_MODEL).invoke(prompt).content.strip()

            if not city:
                return "Could not determine the city from your query."

            params = {
                "engine": "google_maps",
                "q": f"tourist attractions in {city}",
                "type": "search",
                "api_key": os.getenv("SERPAPI_API_KEY"),
            }
            search = GoogleSearch(params)
            results = search.get_dict()

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

            return f"Top sightseeing attractions in {city}: ⭐" + " ".join(attractions)

        except Exception as e:
            return f"[ERROR] FetchSightseeingTool failed: {str(e)}"


class FetchWeatherTool(BaseTool):
    name: str = "fetch_weather"
    description: str = (
        "Get detailed weather forecast for a city & date range. "
        "Input should be a natural language query "
        "(e.g. 'what will the weather be like in Paris from 20th to 25th August')."
    )
    args_schema: Type[BaseModel] = ExtractTripDetailsInput  # single user_query
    trip_memory: TripMemory

    def _run(self, user_query: str) -> Dict[str, List[Dict[str, Any]]]:
        # --- Step 1: Extract city + dates ---
        prompt = f"""
        Extract weather request details from the user query.
        Return ONLY valid JSON in this format:
        {{
          "city": "<city>",
          "start_date": "YYYY-MM-DD",
          "end_date": "YYYY-MM-DD"
        }}

        Rules:
        - Dates must be ISO format (YYYY-MM-DD).
        - If only one date is given, set both start_date and end_date to that date.
        - If year missing, assume next occurrence.

        User: {user_query}
        """

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=60,
            )
            raw = (resp.json() or {}).get("response", "").strip()
            json_str = _extract_json_block(raw)
            details = json.loads(json_str)

            city = details["city"]
            start_date = details["start_date"]
            end_date = details["end_date"]

            # update trip memory
            self.trip_memory.update(details)

        except Exception as e:
            return {"error": f"Failed to parse weather query: {e}"}

        # --- Step 2: Call OpenWeather ---
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        try:
            data = requests.get("https://api.openweathermap.org/data/2.5/forecast", params=params).json()
        except Exception as e:
            return {"error": f"Weather API error: {e}"}

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        except Exception:
            return {"error": "Invalid date format. Use YYYY-MM-DD."}

        # --- Step 3: Filter forecast data ---
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

        return filtered or {"message": f"No weather data found for {city} in that date range."}

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


def build_agent(verbose: bool = True):
    # ---- LLM ----
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        num_ctx=8016,
        temperature=0.2,
        # verbose=verbose,
        num_predict= 8016
    )

    # ---- Trip-level memory ----
    trip_memory = TripMemory()

    # ---- Tools (with shared trip_memory) ----
    tools = [
        SearchFlightsTool( trip_memory=trip_memory),
        SearchHotelsTool(),
        FetchWeatherTool(trip_memory=trip_memory),
        FetchSightseeingTool(),
        CurrencyConvertTool(),
    ]

    # ---- Conversation memory ----
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
        human_prefix="Human",
        ai_prefix="AI",
    )

    # Use the ReAct framework: Think → Act (tool call) → Observe → Answer.

    # ---- ReAct Prefix ----
    prefix = """
    You are TripMate, a Reasoning and Acting Agent. 
    Job is to assist user by answering their queries.
    Paying attention to what actually the user needs.
        Chat history:
        {chat_history}

    Guidelines:
    - When the user asks for new information (e.g. search flights, hotels, weather, sightseeing, currency or multiple at once), call the appropriate tool or tools.
    - If a tool call or calls succeeds and returns data, do not repeat the same tool call.
    - If you are unsure of certain details before making any tool call please ask user for missing details in a follow up question.
    - Start Final answer with your thoughts first then your recomendations on it.
    - Always include the full tool output (formatted text returned by the tool) in your final answer. Do not shorten or drop details.
    - provide the user with a final indepth natural language explanation of the results.
    - Do not repeat the Action or Tool call after successful observation.
    - Never return raw JSON to user.
    - When the user asks for clarification, advice, or preferences based on **previous tool results**, DO NOT call tools again.
    - Instead, recall the last tool or tools outputs from memory and give a clear recommendation.
    - Never drop earlier results (e.g., flights, hotels, weather).
    - Always end with:

    Final Answer: 
    Start Final answer with your thoughts first then your recomendations on it. Make sure it only contains useful information.
    1. First understand the intent (trip planning, flights, hotels, weather, sightseeing, currency or may be the user want multiple answers at once).
    2. If trip planning → call tools in order:
       flights → hotels → weather → sightseeing.
    3. For single-task requests (only hotels, only weather, etc.) → call only that tool.
    4. Reuse context from trip_memory and chat_history for follow-up questions.
    5. Tool calls must be structured JSON. 
    6. Never return raw JSON to user.
    7. Never leave responses blank.
    """


# ---- Build Agent ----
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # verbose=verbose,
        memory=memory,
        max_iterations=5,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        # return_intermediate_steps=True,
        agent_kwargs={
            "prefix": prefix,
            "input_variables": ["input", "chat_history"]
        },
    )

    print("Registered tools:", [t.name for t in tools])
    return agent


if __name__ == "__main__":
    agent = build_agent()
    while True:
        user_input = input("Q> ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "escape"):
            print("👋 Goodbye!")
            break
        try:
            answer = agent.run(user_input)
            if "Final Answer:" not in answer:
                answer = f"Final Answer: {answer}"
                print(answer)
        except Exception as e:
            print(f"[ERROR] {e}")
