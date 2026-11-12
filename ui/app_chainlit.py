"""Module for handling the chat UI interface and streaming backend responses."""

import json
import uuid
from typing import Any

import chainlit as cl
import httpx
from chainlit.logger import logger

from ui.gcp_auth import FASTAPI_CHAT_URL, create_backend_client

NODE_DISPLAY_MAP: dict[str, str] = {
    "input_guard": "🛡️ Memeriksa Keamanan Input Pengguna",
    "context_manager": "🧠 Menganalisis Konteks & Rute Pertanyaan",
    "legal_search": "🔍 Mencari Referensi Pasal Hukum di Database Qdrant",
    "legal_reasoner": "⚖️ Menyusun Analisis & Argumen Hukum",
    "output_guard": "🔬 Memverifikasi Akurasi dan Validitas Jawaban",
}

UU_DISPLAY_MAP: dict[str, str] = {
    "uu_ketenagakerjaan_2003": "UU No. 13 Tahun 2003 tentang Ketenagakerjaan",
    "uu_bpjs_2011": "UU No. 24 Tahun 2011 tentang BPJS",
}


@cl.on_chat_start
async def start_chat() -> None:
    """Initialize the user session and send the initial greeting message."""
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    logger.info(f"[{session_id}] New chat session started by user.")

    greeting = (
        "Halo! Saya adalah **Asisten Hukum Ketenagakerjaan Indonesia**. "
        "Silakan tanyakan masalah ketenagakerjaan Anda."
    )
    await cl.Message(content=greeting).send()


@cl.on_message
async def handle_message(message: cl.Message) -> None:
    """Handle incoming user messages and stream the response from the backend API."""
    session_id = cl.user_session.get("session_id")
    payload = {"message": message.content, "session_id": session_id}

    logger.info(f"[{session_id}] Received user message: {message.content[:50]}...")
    logger.debug(f"[{session_id}] Sending payload to backend: {json.dumps(payload)}")

    msg_placeholder = cl.Message(content="")
    await msg_placeholder.send()

    thinking_step: cl.Step | None = cl.Step(name="AnalisisSistem", type="tool")

    if thinking_step:
        thinking_step.output = "🤖 Asisten Hukum sedang menganalisis regulasi..."
        await thinking_step.send()

    final_answer = ""
    citations: list[dict[str, Any]] = []

    try:
        async with create_backend_client() as http_client:
            logger.info(f"[{session_id}] Opening stream to backend API...")

            async with http_client.stream(
                "POST", FASTAPI_CHAT_URL, json=payload
            ) as response:
                if response.status_code != 200:
                    logger.error(
                        f"[{session_id}] Backend failed with status: "
                        f"{response.status_code}"
                    )
                    if thinking_step:
                        await thinking_step.remove()
                    msg_placeholder.content = (
                        f"❌ Server merespons dengan status eror: "
                        f"{response.status_code}"
                    )
                    await msg_placeholder.update()
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        json_str = line[6:]
                        data = json.loads(json_str)
                        event_type = data.get("event")

                        if event_type == "node_complete":
                            completed_node = data.get("node")
                            logger.info(
                                f"[{session_id}] Node execution completed: "
                                f"{completed_node}"
                            )

                            if thinking_step:
                                if completed_node == "input_guard":
                                    thinking_step.output = (
                                        "🛡️ Keamanan input tervalidasi. "
                                        "Menganalisis konteks hukum..."
                                    )
                                elif completed_node == "context_manager":
                                    thinking_step.output = (
                                        "🧠 Mengekstrak referensi pasal dan "
                                        "memproses database..."
                                    )
                                elif completed_node == "legal_search":
                                    thinking_step.output = (
                                        "🔍 Menyusun argumentasi hukum dan "
                                        "pasal ketenagakerjaan..."
                                    )
                                elif completed_node == "legal_reasoner":
                                    thinking_step.output = (
                                        "⚖️ Memverifikasi validitas akhir dan "
                                        "akurasi jawaban..."
                                    )
                                await thinking_step.update()

                        elif event_type == "final_result":
                            logger.info(
                                f"[{session_id}] Received final_result from backend."
                            )
                            if thinking_step:
                                await thinking_step.remove()
                                thinking_step = None

                            final_answer = data.get("final_answer", "")
                            citations = data.get("citations", [])

                            logger.debug(
                                f"[{session_id}] Final answer content: '{final_answer}'"
                            )
                            logger.debug(
                                f"[{session_id}] Citations content: {citations}"
                            )

                if thinking_step:
                    await thinking_step.remove()

                elements: list[cl.Element] = []
                logger.info(
                    f"[{session_id}] Processing {len(citations)} "
                    f"legal citations for display."
                )

                for cite in citations:
                    raw_uu_id = cite.get("uu_id", "")
                    formatted_uu_id = UU_DISPLAY_MAP.get(raw_uu_id, raw_uu_id)

                    markdown_content = (
                        f"## ⚖️ {formatted_uu_id}\n"
                        f"**{cite.get('bab', '')}**\n\n"
                        f"{cite.get('text_parent', '')}\n\n"
                        f"---\n\n"
                    )

                    elements.append(
                        cl.Text(
                            name="Daftar Sitasi Pasal Terkait",
                            content=markdown_content,
                            display="side",
                        )
                    )

                msg_placeholder.content = final_answer
                if elements:
                    msg_placeholder.elements = elements
                await msg_placeholder.update()

                logger.info(f"[{session_id}] Successfully rendered legal response.")

    except httpx.RequestError as exc:
        logger.error(f"[{session_id}] Network error connecting to backend API: {exc}")

        if thinking_step:
            await thinking_step.remove()
        msg_placeholder.content = f"⚠️ Gagal terhubung ke Backend API Utama: {exc}"
        await msg_placeholder.update()
