"""Latency benchmark against a running Helm server.

    python scripts/bench.py --url http://127.0.0.1:8000 --runs 30

For each tool count it sends one warmup request, then --runs sequential requests, and reports the
server-side latency (latency_ms in the response) and the client wall time. Uses only the standard library."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

TOOL_POOL = [
    ("search_flights", "Search flights between two airports on a given date."),
    ("get_weather_forecast", "Get the daily weather forecast for a city for the next 10 days."),
    ("convert_currency", "Convert an amount of money between two currencies."),
    ("book_hotel", "Book a hotel room in a city for given check-in and check-out dates."),
    ("get_exchange_rates", "Latest exchange rates for a base currency."),
    ("translate_text", "Translate text between languages."),
    ("find_restaurants", "Find restaurants near a location, filtered by cuisine and price."),
    ("get_city_events", "List concerts, shows and events in a city for a date range."),
    ("rent_car", "Reserve a rental car at an airport or city location."),
    ("get_visa_requirements", "Visa and entry requirements for a passport and destination country."),
]
TASK = ("I'm flying from San Francisco to Tokyo next Friday. Find me a flight and tell me what the weather "
        "will be like when I land.")
HISTORY = [{"tool": "search_flights", "args": {"origin": "SFO", "destination": "HND", "date": "2026-10-30"},
            "result": '{"flights": [{"id": "NH7", "arrive": "15:25+1", "price_usd": 1180}]}'}]


def payload(n_tools: int, long_history: bool) -> bytes:
    history = HISTORY * (8 if long_history else 1)
    return json.dumps({"task": TASK, "history": history,
                       "tools": [{"name": n, "description": d} for n, d in TOOL_POOL[:n_tools]]}).encode()


def post(url: str, body: bytes) -> tuple[float, float]:
    req = urllib.request.Request(f"{url}/rank", data=body, headers={"content-type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as res:
        data = json.load(res)
    return data["latency_ms"], 1000.0 * (time.perf_counter() - started)


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--tools", default="2,6,10", help="comma list of tool counts (Finish is added on top)")
    args = parser.parse_args()
    print("tools | history | server p50 / p95 ms | wall p50 ms")
    for n in (int(x) for x in args.tools.split(",")):
        for long_history in (False, True):
            body = payload(n, long_history)
            post(args.url, body)  # warmup
            server, wall = zip(*(post(args.url, body) for _ in range(args.runs)))
            print(f"{n:5d} | {'8 calls' if long_history else '1 call ':7s} | "
                  f"{statistics.median(server):7.0f} / {pct(list(server), 0.95):5.0f}     | {statistics.median(wall):7.0f}")


if __name__ == "__main__":
    main()
