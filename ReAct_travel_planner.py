import os
import json
import requests
from langchain.tools import BaseTool
from typing import Optional, Type, List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime, timedelta
import googlemaps
import time
from amadeus import Client
import re

load_dotenv()
# ===== CONFIG =====

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")

amadeus = Client(
    client_id=AMADEUS_CLIENT_ID,
    client_secret=AMADEUS_CLIENT_SECRET
)

# OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_MODEL = "gpt-oss:20b"

llm = OllamaLLM(OLLAMA_MODEL)


DAILY_LIMIT = 330  # Google Maps quota

_calls_made = {"places": 0}

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# ===== LIGHTWEIGHT MEMORY (very small and optional) =====
MEMORY_FILE = "travel_memory.json"

def load_memory() -> Dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_memory(memory: Dict):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
    except:
        pass

def update_memory(kv: Dict):
    mem = load_memory()
    mem.update({k: v for k, v in kv.items() if v})
    save_memory(mem)


# ===== INTENT DETECTION =====
def detect_user_intent(user_query: str) -> Dict:
    """
    Uses LLM to detect what the user actually wants to do.
    Returns intent type and required parameters.
    """
    prompt = f"""
    You are an AI assistant that detects user intent for travel-related queries.

    Analyze the user's query and determine their primary intent. Return ONLY valid JSON with this structure:

    {{
        "intent": "<primary_intent>",
        "confidence": <0.0-1.0>,
        "requires_followup": <true/false>,
        "extracted_params": {{
            // any parameters you can extract from the query
        }},
        "missing_params": ["list", "of", "required", "params", "still", "needed"]
    }}

    Possible intents:
    - "weather": User wants weather information for a city/destination
    - "currency": User wants currency conversion or exchange rates
    - "flights": User wants flight search/information only
    - "hotels": User wants accommodation information only  
    - "attractions": User wants tourist attractions/sightseeing info
    - "full_trip": User wants complete trip planning
    - "general_info": User wants general travel information about a place

    Examples:
    - "What's the weather like in Paris?" → intent: "weather"
    - "Convert 500 USD to INR" → intent: "currency"  
    - "Show me flights from NYC to London" → intent: "flights"
    - "Plan a trip to Rome for 4 people" → intent: "full_trip"
    - "What are the top attractions in Tokyo?" → intent: "attractions"

    User query: "{user_query}"
    """

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        )
        raw_text = resp.json()["response"].strip()

        # Extract JSON from response
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}") + 1
        json_str = raw_text[start_idx:end_idx]

        return json.loads(json_str)
    except Exception as e:
        # Fallback if LLM fails
        return {
            "intent": "general_info",
            "confidence": 0.5,
            "requires_followup": True,
            "extracted_params": {},
            "missing_params": []
        }


# ===== PARAMETER EXTRACTION FOR SPECIFIC INTENTS =====
def extract_intent_params(user_query: str, intent: str) -> Dict:
    """Extract specific parameters based on the detected intent."""

    if intent == "weather":
        prompt = f"""
        Extract weather-related parameters from: "{user_query}"
        Return JSON: {{"city": "<city_name>", "start_date": "YYYY-MM-DD or null", "end_date": "YYYY-MM-DD or null"}}
        If dates are missing, set to null.
        """
    elif intent == "currency":
        prompt = f"""
        Extract currency conversion parameters from: "{user_query}"
        Return JSON: {{"amount": <number or null>, "from_currency": "<currency_code or null>", "to_currency": "<currency_code or null>"}}
        """
    elif intent == "flights":
        prompt = f"""
        Extract flight search parameters from: "{user_query}"
        Return JSON: {{"origin": "<city/IATA or null>", "destination": "<city/IATA or null>", "start_date": "YYYY-MM-DD or null", "end_date": "YYYY-MM-DD or null", "adults": <number or null>}}
        """
    elif intent == "hotels":
        prompt = f"""
        Extract hotel search parameters from: "{user_query}"
        Return JSON: {{"destination_city": "<city or null>", "check_in_date": "YYYY-MM-DD or null", "check_out_date": "YYYY-MM-DD or null", "adults": <number or null>}}
        """
    elif intent == "attractions":
        prompt = f"""
        Extract attraction search parameters from: "{user_query}"
        Return JSON: {{"city": "<city_name or null>"}}
        """
    else:
        # For full_trip or general_info, use the original extraction
        return extract_travel_details(user_query)

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        )
        raw_text = resp.json()["response"].strip()

        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}") + 1
        json_str = raw_text[start_idx:end_idx]

        return json.loads(json_str)
    except:
        return {}


# ===== SMART FOLLOW-UP FOR MISSING PARAMS =====
def ask_for_missing_params(intent: str, extracted_params: Dict, missing_params: List[str]) -> Dict:
    """Ask user for missing parameters specific to their intent."""

    if not missing_params:
        return extracted_params

    # Create intent-specific prompts
    followup_prompts = {
        "weather": {
            "city": "Which city would you like the weather forecast for?",
            "start_date": "For which date? (or starting when?)",
            "end_date": "Until which date? (optional, press enter to skip)"
        },
        "currency": {
            "amount": "How much would you like to convert?",
            "from_currency": "From which currency?",
            "to_currency": "To which currency?"
        },
        "flights": {
            "origin": "From which city/airport?",
            "destination": "To which destination?",
            "start_date": "What's your departure date?",
            "end_date": "What's your return date? (optional for one-way)",
            "adults": "How many travelers?"
        },
        "hotels": {
            "destination_city": "In which city are you looking for hotels?",
            "check_in_date": "What's your check-in date?",
            "check_out_date": "What's your check-out date?",
            "adults": "How many guests?"
        },
        "attractions": {
            "city": "Which city's attractions would you like to explore?"
        }
    }

    current_prompts = followup_prompts.get(intent, {})

    for param in missing_params:
        if param in current_prompts:
            print(current_prompts[param])
            answer = input("> ").strip()

            if answer:  # Only update if user provided an answer
                # Use LLM to intelligently parse the answer
                parse_prompt = f"""
                The user answered "{answer}" for the parameter "{param}".
                Parse this into the appropriate format:
                - For dates: return YYYY-MM-DD format
                - For numbers: return integer
                - For text: return as-is
                Return only the parsed value, nothing else.
                """

                try:
                    resp = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": OLLAMA_MODEL, "prompt": parse_prompt, "stream": False}
                    )
                    parsed_value = resp.json()["response"].strip().strip('"')
                    extracted_params[param] = parsed_value
                except:
                    extracted_params[param] = answer

    return extracted_params

# ===== SMALL city->IATA fallback to keep your flow working =====
def get_iata_code_with_llm(llm, city_name: str):
    """
    Uses the LLM to determine the IATA airport code for a given city name.
    If ambiguous, the LLM should request clarification from the user.
    """

    prompt = f"""
    You are a travel planning assistant.
    Given the city name below, return ONLY the most likely IATA airport code.
    If there are multiple main airports in that city, say 'AMBIGUOUS' and list them.
    If the city name is unclear, say 'UNKNOWN'.

    City: {city_name}

    Example output:
    - "FCO"
    - "AMBIGUOUS: NRT, HND"
    - "UNKNOWN"
    """

    response = llm.ask(prompt).strip()
    return response


# ===== SYSTEM PROMPT (ReAct + follow-ups + budget-first) =====
SYSTEM_PROMPT = """
You are TripSmith — a ReAct (reason + act) AI travel planner with memory.
Your job is to plan trips that fit the user's budget, using tools only after you have all key details.
You've been given few tools tools must only be called after understanding user's intent first. User may not always ask for detailed trip plans.
may only be asking for weather, currency exchange rates, flight options, about stays/ accommodations, may ask for tourist attractions. 
you have to decide the tool accordingly

STRICT RULES:
1) BEFORE ANY TOOL CALLS, ask the user for ALL missing details:
   - origin (IATA), destination (IATA) and destination_city
   - exact dates (start_date and end_date, YYYY-MM-DD). If missing: ask "Do you have any specific dates in mind?"
   - number of travelers (ask "How many travelers?")
   - total budget and currency (e.g., "2500 USD")
   - preferred currency for results
   If any of these are missing or ambiguous, ask follow-up questions first.

2) Use natural language understanding — avoid regex-heavy parsing. 
Interpret amounts like "2.5k", "about 2500$", "three thousand dollars", date phrases like "in May", and counts like "for 4 people".

3) Respect budget. If constraints look over budget, 
explain clearly and suggest options (shift dates, nearby airports, fewer nights, cheaper hotels).

4) Keep responses concise and actionable. After acting with tools,
 summarize and ask if the user wants refinements.

5) Memory: remember stable preferences (home airport, preferred currency,
typical party size/budget) for future runs. Never invent. 

You must ALWAYS ask for missing details before any tool call.
"""

# ===== STEP 1 — Extract travel details (LLM-first; adds adults & budget) =====
def extract_travel_details(user_query: str):
    memory = load_memory()
    prompt = f"""
{SYSTEM_PROMPT}

Task: From the user's text, extract structured details. Use your natural language understanding.
If a field is missing, set it to null (do NOT guess). If a city (not IATA) is provided, keep it as-is; the program may map to IATA later.
Return ONLY valid JSON with these keys:

Rules:
1. Identify origin city and destination city, in **any word order**.
   - Look for “from X to Y”, “to Y from X”, “X → Y”, “flying into Y from X”.
   - If only a country is given, pick the busiest international airport in that country.
   - Always convert to the correct 3-letter IATA airport code.

2. Identify dates:
   - Understand exact dates (“22nd August 2025”), month + day (“Aug 22”), or relative phrases (“next Monday”, “in 2 weeks”).
   - Handle ranges (“Aug 22 to Aug 27”), trip lengths (“for 5 days”), seasons (“this winter”), holidays (“Christmas”).
   - If the year is missing, assume the next occurrence from today.
   - If only start date and trip length are given, calculate the return date.

3. Identify number of travelers if mentioned, else default to 2

{{
  "origin": "<IATA or city or null>",
  "destination": "<IATA or city or null>",
  "destination_city": "<city or null>",
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "adults": <int or null>,
  "budget": {{"amount": <number or null>, "currency": "<str or null>"}},
  "currency": "<pricing currency or null>"
}}

if any of those details are missing, ask user with the follow up question.
Consider memory (if helpful defaults exist): {json.dumps(memory)}

User:
\"\"\"{user_query}\"\"\"
"""
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    raw_text = resp.json()["response"].strip()

    # Try to isolate JSON if LLM returns extra text
    start_idx = raw_text.find("{")
    end_idx = raw_text.rfind("}") + 1
    json_str = raw_text[start_idx:end_idx]

    try:
        details = json.loads(json_str)
    except json.JSONDecodeError:
        details = {
            "origin": None, "destination": None, "destination_city": None,
            "start_date": None, "end_date": None, "adults": None,
            "budget": {"amount": None, "currency": None},
            "currency": None
        }
    return details
def ask_missing(details: dict) -> dict:
    """
    Uses the LLM to identify missing trip details and ask follow-up questions
    in a natural, conversational way until all required info is collected.
    """

    while True:
        # Step 1 — Ask LLM what’s missing & get a natural follow-up
        prompt = f"""
        You are a friendly AI travel agent. The user has provided the following details so far:
        {json.dumps(details, indent=2)}

        Required fields: origin city, destination city, start date, end date,
        number of travelers (adults), budget amount, budget currency.

        Your task:
        1. Identify which fields are missing or unclear.
        2. If *all* required fields are present and clear, reply only with: "DONE".
        3. Otherwise, return JSON:
           {{
             "missing_fields": ["list", "of", "fields"],
             "followup_question": "natural question to ask to get next missing info"
           }}
        4. The follow-up question should be warm and natural, not a rigid form prompt.
        5. If asking for dates, accept flexible answers like "next summer" and clarify later if needed.
        """

        llm_res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        ).json()["response"].strip()

        if llm_res.upper().startswith("DONE"):
            break

        try:
            parsed = json.loads(llm_res)
        except json.JSONDecodeError:
            print("LLM response error, retrying...")
            continue

        # Step 2 — Ask user the follow-up question
        print(parsed["followup_question"])
        answer = input("> ").strip()

        # Step 3 — Let LLM update the details dictionary from the answer
        update_prompt = f"""
        You are updating the user's trip details.
        Current details:
        {json.dumps(details, indent=2)}

        The user responded: "{answer}"

        Update the dictionary with any fields that can be inferred from the answer.
        For example:
        - Convert city names to IATA codes if possible
        - Parse dates if they are exact (YYYY-MM-DD) or infer an approximate period
        - Extract number of travelers, budget amount, and currency if provided
        Return ONLY valid JSON of the updated details.
        """

        updated_res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": update_prompt, "stream": False}
        ).json()["response"].strip()

        try:
            details = json.loads(updated_res)
        except json.JSONDecodeError:
            print("LLM update error, skipping update.")
            continue

    # Step 4 — Persist light preferences
    update_memory({
        "home_airport": details.get("origin"),
        "preferred_currency": details.get("currency"),
        "default_adults": details.get("adults"),
        "typical_budget": details.get("budget")
    })

    return details


# ===== STEP 3 — Tool Schemas =====
class FlightSearchInput(BaseModel):
    origin: str = Field(..., description="Origin city name or IATA code of departure airport (e.g., 'New York' or 'JFK')")
    destination: str = Field(..., description="Destination city name or IATA code of arrival airport (e.g., 'Rome' or 'FCO')")
    start_date: str = Field(..., description="Departure date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="Return date in YYYY-MM-DD format (optional)")
    # currency: str = Field(default="USD", description="Currency code for pricing")
    adults: int = Field(default=1, description="Number of adult travelers")

# ===== STEP 4 — Tools =====
class SearchFlightsTool(BaseTool):
    name: str = "search_flights"
    description: str = "Search for flights using the Amadeus API and return structured results."
    args_schema: Type[BaseModel] = FlightSearchInput

    def _run(self, origin: str, destination: str, start_date: str, end_date: Optional[str] = None,
             # currency: str = "USD",
             adults: int = 1) -> List[Dict]:
        try:
            # ✅ Resolve IATA codes with LLM
            origin_code = get_iata_code_with_llm(llm, origin)
            destination_code = get_iata_code_with_llm(llm, destination)

            # Handle ambiguous/unknown cases
            if origin_code.startswith("AMBIGUOUS") or origin_code == "UNKNOWN":
                return [{"error": f"Origin city ambiguous or unknown: {origin_code}"}]
            if destination_code.startswith("AMBIGUOUS") or destination_code == "UNKNOWN":
                return [{"error": f"Destination city ambiguous or unknown: {destination_code}"}]

            params = {
                "originLocationCode": origin_code,
                "destinationLocationCode": destination_code,
                "departureDate": start_date,
                "adults": adults,
                # "currencyCode": currency,
                "max": 5
            }
            if end_date:
                params["returnDate"] = end_date

            response = amadeus.shopping.flight_offers_search.get(**params)
            data = response.data

            def format_duration(iso_duration):
                match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", iso_duration)
                if not match:
                    return iso_duration
                hours = match.group(1) or "0"
                minutes = match.group(2) or "0"
                return f"{int(hours)}h {int(minutes)}m"

            def format_time(iso_time):
                try:
                    return datetime.fromisoformat(iso_time).strftime("%Y-%m-%d %H:%M")
                except:
                    return iso_time

            def format_segments(segments):
                legs = []
                for seg in segments:
                    dep_airport = seg['departure']['iataCode']
                    arr_airport = seg['arrival']['iataCode']
                    dep_time = format_time(seg['departure']['at'])
                    arr_time = format_time(seg['arrival']['at'])
                    duration = format_duration(seg['duration'])
                    legs.append(f"{dep_airport} ({dep_time}) → {arr_airport} ({arr_time}) [{duration}]")
                return legs

            results = []
            for offer in data:
                price = f"{offer['price']['total']} {offer['price']['currency']}"
                airline = offer["validatingAirlineCodes"][0]
                outbound_legs = format_segments(offer["itineraries"][0]["segments"])
                return_legs = format_segments(offer["itineraries"][1]["segments"]) if len(
                    offer["itineraries"]) > 1 else []
                results.append({
                    "price": price,
                    "airline": airline,
                    "outbound": outbound_legs,
                    "return": return_legs
                })
            return results

        except Exception as e:
            return [{"error": str(e)}]

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not implemented for SearchFlightsTool")


class HotelSearchInput(BaseModel):
    destination_city: str = Field(..., description="City name of the destination")
    check_in_date: str = Field(..., description="Hotel check-in date in YYYY-MM-DD format")
    check_out_date: str = Field(..., description="Hotel check-out date in YYYY-MM-DD format")
    adults: int = Field(default=2, description="Number of adults")


class SearchHotelsTool(BaseTool):
    name: str = "search_hotels"
    description: str = "Find hotels in a given destination city."
    args_schema: Type[BaseModel] = HotelSearchInput

    def _run(self, destination_city: str, check_in_date: str, check_out_date: str, adults: int = 2):
        params = {
            "engine": "google_hotels",
            "q": destination_city,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "currency": "USD",
            "gl": "us",
            "hl": "en",
            "api_key": SERPAPI_API_KEY
        }

        resp = requests.get("https://serpapi.com/search", params=params)
        resp.raise_for_status()
        data = resp.json()

        hotels = []
        if "properties" in data:
            for h in data["properties"]:
                hotels.append({
                    "name": h.get("name"),
                    "price": h.get("rate_per_night", {}).get("lowest"),
                    "rating": h.get("overall_rating"),
                    "address": h.get("address"),
                    "link": h.get("link")
                })
        return hotels

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not implemented for SearchHotelsTool")


class CurrencyInput(BaseModel):
    amount: float = Field(..., description="Amount to convert")
    from_currency: str = Field(..., description="Source currency")
    to_currency: str = Field(default="INR", description="Target currency")

class CurrencyConvertTool(BaseTool):
    name: str = "convert_currency"
    description: str = "Convert currency amounts"
    args_schema: Type[BaseModel] = CurrencyInput

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


class SightseeingInput(BaseModel):
    city: str = Field(..., description="City to search for tourist attractions.")


class FetchSightseeingTool(BaseTool):
    name: str = "fetch_sightseeing"
    description: str = "Get top tourist attractions for a given city."
    args_schema: Type[BaseModel] = SightseeingInput

    def _run(self, city: str):
        if _calls_made["places"] >= DAILY_LIMIT:
            return [{"error": "Google Maps API daily limit reached"}]

        try:
            places = gmaps.places(query=f"tourist attractions in {city}")
            _calls_made["places"] += 1
            results = []
            for p in places.get("results", []):
                results.append({
                    "name": p.get("name"),
                    "rating": p.get("rating"),
                    "address": p.get("formatted_address"),
                    "types": p.get("types", []),
                })
            return results
        except Exception as e:
            return [{"error": str(e)}]

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


class WeatherInput(BaseModel):
    city: str = Field(..., description="City name for weather forecast.")
    start_date: str = Field(..., description="Trip start date in YYYY-MM-DD format.")
    end_date: str = Field(..., description="Trip end date in YYYY-MM-DD format.")


class FetchWeatherTool(BaseTool):
    name: str = "fetch_weather"
    description: str = "Get detailed weather forecast for a given city and date range."
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str, start_date: str, end_date: str):
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        data = requests.get("https://api.openweathermap.org/data/2.5/forecast", params=params).json()

        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

        filtered_forecast = {}
        for entry in data.get("list", []):
            entry_dt = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
            entry_date = entry_dt.date()

            if start_dt <= entry_date <= end_dt:
                day_str = entry_date.strftime("%Y-%m-%d")
                filtered_forecast.setdefault(day_str, []).append({
                    "time": entry_dt.strftime("%H:%M"),
                    "temp": round(entry["main"]["temp"], 1),
                    "feels_like": round(entry["main"]["feels_like"], 1),
                    "condition": entry["weather"][0]["description"].title(),
                    "humidity": entry["main"]["humidity"],
                    "wind_speed": entry["wind"]["speed"]
                })

        return filtered_forecast

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


def execute_intent(intent: str, params: Dict):
    """Execute the appropriate tool based on user intent."""

    if intent == "weather":
        tool = FetchWeatherTool()
        return tool.run(params)

    elif intent == "currency":
        tool = CurrencyConvertTool()
        return tool.run(params)

    elif intent == "flights":
        tool = SearchFlightsTool()
        return tool.run(params)

    elif intent == "attractions":
        tool = FetchSightseeingTool()
        return tool.run(params)

    elif intent == "full_trip":
        # Run multiple tools for comprehensive planning
        return execute_full_trip_planning(params)

    else:
        return {
            "message": "I can help with weather, currency conversion, flights, hotels, attractions, or full trip planning. What would you like to know?"}


def execute_full_trip_planning(params: Dict):
    """Execute full trip planning with multiple tools."""
    results = {}

    # Search flights if origin/destination provided
    if params.get("origin") and params.get("destination"):
        flights_tool = SearchFlightsTool()
        results["flights"] = flights_tool.run({
            "origin": params["origin"],
            "destination": params["destination"],
            "start_date": params.get("start_date"),
            "end_date": params.get("end_date"),
            "adults": params.get("adults", 2)})

    # Get weather if destination city provided
    if params.get("destination_city"):
        weather_tool = FetchWeatherTool()
        results["weather"] = weather_tool.run({
            "city": params["destination_city"],
            "start_date": params.get("start_date"),
            "end_date": params.get("end_date")})

        # Get attractions
        attractions_tool = FetchSightseeingTool()
        results["attractions"] = attractions_tool.run({
            "city": params["destination_city"]})

        if params.get("start_date") and params.get("end_date"):
            hotels_tool = SearchHotelsTool()
            results["hotels"] = hotels_tool.run({
                "city": params["destination_city"],
                "check_in": params["start_date"],
                "check_out": params["end_date"],
                "adults": params.get("adults", 2)
            })

    return results



# ===== STEP 5 — Main agent =====
def run_agent():
    print("🌍 TripSmith Assistant - I can help with weather, flights, currency, attractions, or full trip planning!")
    while True:
        user_query = input("\nWhat can I help you with? (or 'quit' to exit): ").strip()

        if user_query.lower() in ['quit', 'exit', 'bye']:
            print("Safe travels! 🛫")
            break

        # Step 1: Detect user intent
        print("🔍 Understanding your request...")
        intent_data = detect_user_intent(user_query)
        intent = intent_data.get("intent")
        confidence = intent_data.get("confidence", 0.0)
        print(f"📋 Detected intent: {intent} (confidence: {confidence:.1f})")

        # Step 2: Extract parameters
        params = extract_intent_params(user_query, intent)

        # Step 3: Ask for missing parameters if needed
        missing_params = intent_data.get("missing_params", [])
        if missing_params:
            print("I need a bit more information...")
            params = ask_for_missing_params(intent, params, missing_params)

        # Step 4: Execute the intent
        print(f"⚡ Executing {intent} request...")
        results = execute_intent(intent, params)

        # Step 5: Present results
        print("\n" + "=" * 50)
        present_results(intent, results, params)
        print("=" * 50)


if __name__ == "__main__":
    # Example the agent will handle with follow-ups:
    # "I have 2500$ for a trip to rome from Toronto in May for 4 people."
    # It will ask: specific dates & any missing items before tool calls.
    run_agent()
