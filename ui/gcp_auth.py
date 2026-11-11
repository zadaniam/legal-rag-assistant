"""Google Cloud Run authentication utilities for backend API requests."""

from collections.abc import Generator

import httpx
from chainlit.logger import logger

from ui.config import ui_settings

FASTAPI_CHAT_URL: str = f"{ui_settings.BACKEND_URL}/api/v1/chat"


class GoogleCloudRunAuth(httpx.Auth):
    """Custom auth flow that injects Google Cloud Identity Tokens.

    The token is only attached when running in production on Cloud Run. In
    other environments the request proceeds without authentication headers
    to avoid unnecessary metadata server calls during local development.
    """

    requires_response_body = False

    def __init__(self, target_url: str) -> None:
        self.target_audience = target_url.replace("/api/v1/chat", "")
        self.metadata_endpoint = (
            "http://metadata.google.internal/computeMetadata/v1"
            "/instance/service-accounts/default/identity"
        )

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Yield the request after optionally attaching a Google Identity Token.

        The token is fetched from the Cloud Run metadata server only when
        APP_ENV is production. If the metadata server is unreachable, the
        request continues without a token to avoid blocking non-production
        environments.
        """
        if ui_settings.ENVIRONMENT == "production":
            headers = {"Metadata-Flavor": "Google"}
            params = {"audience": self.target_audience}

            try:
                with httpx.Client(timeout=2.0) as client:
                    response = client.get(
                        self.metadata_endpoint, headers=headers, params=params
                    )
                    if response.status_code == 200:
                        token = response.text
                        request.headers["Authorization"] = f"Bearer {token}"
            except Exception as e:
                logger.warning(
                    f"Google metadata server unreachable, proceeding without token: {e}"
                )

        yield request


def create_backend_client() -> httpx.AsyncClient:
    """Create an HTTP client configured with Google Cloud Run authentication.

    The client automatically attaches an Identity Token when running in
    production, and connects to the FastAPI chat endpoint.
    """
    return httpx.AsyncClient(
        timeout=60.0,
        auth=GoogleCloudRunAuth(FASTAPI_CHAT_URL),
    )
