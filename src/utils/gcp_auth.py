# src/utils/gcp_auth.py

import os
import httpx

APP_ENV = os.getenv("APP_ENV", "development").lower()
FASTAPI_CHAT_URL = os.getenv("FASTAPI_CHAT_URL", "http://127.0.0.1:7000/api/v1/chat")

class GoogleCloudRunAuth(httpx.Auth):
    """
    Custom Auth Flow untuk menyuntikkan Identity Token Google Cloud secara otomatis
    hanya jika berjalan di lingkungan produksi (Cloud Run).
    """
    def __init__(self, target_url: str):
        # Ambil base URL (target audience) tanpa path API
        self.target_audience = target_url.replace("/api/v1/chat", "")
        self.metadata_endpoint = (
            "http://metadata.google.internal/computeMetadata/v1"
            "/instance/service-accounts/default/identity"
        )
    def requires_response_body(self, request: httpx.Request) -> bool:
        return False

    def auth_flow(self, request: httpx.Request):
        # Hanya jalankan jika di production lingkungan Cloud Run
        if APP_ENV == "production":
            headers = {"Metadata-Flavor": "Google"}
            params = {"audience": self.target_audience}
            
            try:
                # Menggunakan client sinkron internal khusus untuk mengambil token metadata
                with httpx.Client(timeout=2.0) as client:
                    response = client.get(self.metadata_endpoint, headers=headers, params=params)
                    if response.status_code == 200:
                        token = response.text
                        request.headers["Authorization"] = f"Bearer {token}"
            except Exception:
                pass # Tetap lanjutkan request tanpa token jika metadata server gagal dijangkau
                
        yield request

# Factory function untuk menghasilkan HTTP Client yang sudah siap pakai
def create_backend_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=60.0,
        auth=GoogleCloudRunAuth(FASTAPI_CHAT_URL) # Autentikasi disuntikkan secara pasif di sini!
    )
