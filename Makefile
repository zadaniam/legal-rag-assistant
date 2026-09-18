run-app-dev:
	APP_ENV=development uv run uvicorn src.main:app --port 7000 --reload

run-app-prod:
	APP_ENV=production uv run uvicorn src.main:app --reload

run-ui-dev:
	uv run python -m chainlit run ui/app_chainlit.py --port 8000

init-qdrant-prod:
	APP_ENV=production uv run python -m src.database.init_qdrant

ingest-qdrant-prod:
	APP_ENV=production uv run python -m ingestion.mock.run_mock_ingest



