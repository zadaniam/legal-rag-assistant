up-dev:
	docker-compose -f docker-compose.dev.yml up -d

down-dev:
	docker-compose -f docker-compose.dev.yml down

run-app-dev:
	APP_ENV=development uv run uvicorn src.main:app --port 7000 --reload

run-ui-dev:
	APP_ENV=development uv run python -m chainlit run ui/app_chainlit.py --port 8000

init-qdrant-prod:
	APP_ENV=production uv run python -m src.database.init_qdrant

ingest-qdrant-prod:
	APP_ENV=production uv run python -m ingestion.mock.run_mock_ingest