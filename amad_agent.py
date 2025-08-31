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

OLLAMA_MODEL = "llama3.1:8b"

DAILY_LIMIT = 330  # Google Maps quota
_calls_made = {"places": 0}

# GOOGLE_MAPS_API_KEY = "your_gmaps_key"
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


# ===== STEP 1 — Extract travel details =====
def extract_travel_details(user_query: str):
    prompt = f"""
You are a highly accurate travel assistant.
Your job: Extract structured travel details from a user's request and return **only** valid JSON.

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

3. Identify number of travelers if mentioned, else default to 2.

4. Output must be:
{{
  "origin": "<IATA code>",
  "destination": "<IATA code>",
  "destination_city": "<city name>",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "adults": 2
}}

5. Weather Integration:
   - Ensure "destination_city", "start_date", and "end_date" are always filled in.
   - Dates must cover the exact trip duration so weather forecast can be fetched for each day.

Example 1:
User: "to Paris from Mumbai on 22nd of August 2025 for 5 days"
Output: {{"origin": "BOM", "destination_city": "Paris", "destination": "CDG", "start_date": "2025-08-22", "end_date": "2025-08-27", "adults": 2}}

Example 2:
User: "fly from JFK to Dubai next Tuesday for a week"
Output: {{"origin": "JFK", "destination_city": "Dubai","destination": "DXB", "start_date": "2025-08-12", "end_date": "2025-08-19", "adults": 2}}

Example 3:
User: "We want to go to Rome from NYC on June 5 for 7 days"
Output: {{"origin": "JFK","destination_city": "Rome", "destination": "FCO","start_date": "2025-06-05","end_date": "2025-06-12", "adults": 2}}

Example 4:
User: I’m planning to fly from Los Angeles to Tokyo this November 10 for 8 days with my wife, don't look for hotel as we'd be staying at our friends' place.
Output: {{"origin": "LAX","destination_city": "Tokyo", "destination": "NTR","start_date": "2025-11-10","end_date": "2025-11-18", "adults": 2}}

Now process:
"{user_query}"
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
        details = {"origin": None, "destination": None, "start_date": None, "end_date": None}

    return details


# ===== STEP 3 — Tool Schemas =====

class FlightSearchInput(BaseModel):
    origin: str = Field(..., description="IATA code of departure airport")
    destination: str = Field(..., description="IATA code of arrival airport")
    start_date: str = Field(..., description="Departure date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="Return date in YYYY-MM-DD format (optional)")
    currency: str = Field(default="USD", description="Currency code for pricing")

# ===== STEP 4 — Tools =====
class SearchFlightsTool(BaseTool):
    name: str = "search_flights"
    description: str = "Search for flights using the Amadeus API and return structured results."
    args_schema: Type[BaseModel] = FlightSearchInput

    def _run(self, origin: str, destination: str, start_date: str, end_date: Optional[str] = None,
             currency: str = "USD") -> List[Dict]:
        try:
            params = {
                "originLocationCode": origin,
                "destinationLocationCode": destination,
                "departureDate": start_date,
                "adults": 1,
                "currencyCode": currency,
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

class CurrencyConversionInput(BaseModel):
    amount: float = Field(..., description="The amount of money to convert.")
    from_currency: str = Field(..., description="The currency code to convert from (e.g., 'USD').")

class ConvertToINRTool(BaseTool):
    name: str = "convert_to_inr"
    description: str = "Convert an amount from any currency to INR."
    args_schema: Type[BaseModel] = CurrencyConversionInput

    def _run(self, amount: float, from_currency: str):
        if from_currency.upper() == "INR":
            return round(amount, 2)
        params = {
            "apikey": CURRENCY_API_KEY,
            "base_currency": from_currency,
            "currencies": "INR"
        }
        data = requests.get("https://api.currencyapi.com/v3/latest", params=params).json()
        rate = data["data"]["INR"]["value"]
        return round(amount * rate, 2)

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
            raise RuntimeError("Daily Places-API limit reached")
        _calls_made["places"] += 1
        time.sleep(0.1)
        results = gmaps.places(query=f"tourist attractions in {city}").get("results", [])[:5]
        return [
            {"name": p["name"], "address": p.get("formatted_address"), "rating": p.get("rating")}
            for p in results
        ]

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


# ===== STEP 5 — Main agent =====
def run_agent():
    user_query = input("Where do you want to travel? ")

    details = extract_travel_details(user_query)
    print("\nExtracted travel details:", details)

    # Auto-convert city names to IATA if needed
    if details["origin"] and len(details["origin"]) != 3:
        details["origin"] = city_to_iata(details["origin"])
    if details["destination"] and len(details["destination"]) != 3:
        details["destination"] = city_to_iata(details["destination"])

    # Ask user if any critical detail missing
    if not details["origin"]:
        details["origin"] = input("Enter origin IATA code: ").strip().upper()
    if not details["destination"]:
        details["destination"] = input("Enter destination IATA code: ").strip().upper()
    if not details["start_date"]:
        details["start_date"] = input("Enter departure date (YYYY-MM-DD): ").strip()
    if not details["end_date"]:
        details["end_date"] = input("Enter return date (YYYY-MM-DD): ").strip()

    # Search flights with SerpAPI
    print("\nSearching flights...")
    flights_tool = SearchFlightsTool()
    flights_resp = flights_tool.run({
        "origin": details["origin"],
        "destination": details["destination"],
        "start_date": details["start_date"],
        "end_date": details["end_date"],
        "currency": "USD"
    })

    print("\nSearching hotels...")
    hotels_tool = SearchHotelsTool()
    hotels_resp = hotels_tool.run({
        "destination_city": details["destination_city"],
        "check_in_date": details["start_date"],
        "check_out_date": details["end_date"],
        "adults": 2})

    weather_tool = FetchWeatherTool()
    # Fetch weather forecast
    print("\n☀️ Fetching weather forecast...")
    weather_resp = weather_tool.run({
        "city": details["destination_city"],
        "start_date": details["start_date"],
        "end_date": details["end_date"]
    })

    currency_tool = ConvertToINRTool()

    sightseeing_tool = FetchSightseeingTool()
    sightseeing = sightseeing_tool.run({
        "city": details["destination_city"]
    })

    if flights_resp and "error" not in flights_resp[0]:
        print(f"Found {len(flights_resp)} flight options.")
        for f in flights_resp:
            print(f"\n💰 Price: {f['price']} | ✈ Airline: {f['airline']}")
            print("  Outbound:")
            for leg in f['outbound']:
                print(f"    {leg}")
            if f['return']:
                print("  Return:")
                for leg in f['return']:
                    print(f"    {leg}")
    else:
        print("No flights found or an error occurred:", flights_resp)

    print(f"\nFound {len(hotels_resp)} hotel options.")
    for h in hotels_resp[:5]:
        print(f"\nName: {h.get('name')}")
        print(f"Price per night: {h.get('price')}")
        print(f"Rating: {h.get('rating')}")
        print(f"Address: {h.get('address')}")
        print(f"Link: {h.get('link')}")

    # Print weather forecast
    # Weather forecast
    print(f"\n🌦 Weather forecast for {details['destination_city']} ({details['start_date']} → {details['end_date']}):")
    # print(weather_resp)
    for date, readings in weather_resp.items():
        # Daily summary
        temps = [r['temp'] for r in readings]
        print(f"\n📅 {date} | 🌡 {min(temps)}°C – {max(temps)}°C | {readings[0]['condition']}")
        # Detailed 3-hourly
        for r in readings:
            print(f"  ⏰ {r['time']} | 🌡 {r['temp']}°C (feels {r['feels_like']}°C) | "
                  f"{r['condition']} | 💧 {r['humidity']}% | 💨 {r['wind_speed']} m/s")

    print(f"\n🏛 Sites to visit in {details['destination_city']}:")
    for place in sightseeing:
        print(f"  • {place['name']} ({place.get('rating', 'N/A')}⭐) — {place['address']}")


if __name__ == "__main__":
    run_agent()
