# Beavies Surf Forecast

FastAPI prototype for surf spots and forecasts. Weather and marine requests are sent to their respective Open-Meteo endpoints, validated and joined on UTC timestamps. Wind is explicitly requested in metres per second to match the scoring function. Missing hours remain null; zero swell is preserved. Responses include source, units, forecast type and retrieval timestamp.

Run `uvicorn app:app` and inspect `/docs`. Tests: `python -m unittest discover -v`. The score is a heuristic, not a trained machine-learning prediction or a safety recommendation.

Commercial blockers: authenticated users and ownership checks; a durable database instead of in-memory stores; validated preference ranges; cache/rate limits; forecast-quality backtesting; commercial provider terms and subscription configuration. No paid service readiness is claimed. Provider documentation: https://open-meteo.com/en/docs/marine-weather-api and https://open-meteo.com/en/docs .
