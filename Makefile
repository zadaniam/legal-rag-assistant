run-app-dev:
	APP_ENV=development uv run uvicorn src.main:app --port 7000 --reload

run-app-prod:
	APP_ENV=production uv run uvicorn src.main:app --reload

run-ui-dev:
	uv run chainlit run app_chainlit.py --port 8000

init-db-prod:
	APP_ENV=production uv run python -m src.database.init_db

ingest-prod:
	APP_ENV=production uv run python -m src.database.run_mock_ingest



