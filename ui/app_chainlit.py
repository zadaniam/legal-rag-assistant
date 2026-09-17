# app_chainlit.py

import json
import os
import uuid

import chainlit as cl
import httpx

from src.utils.gcp_auth import FASTAPI_CHAT_URL, create_backend_client

# Mapping nama Node LangGraph ke Bahasa Indonesia agar tampilan UI profesional
NODE_DISPLAY_MAP = {
    "input_guard": "🛡️ Memeriksa Keamanan Input Pengguna",
    "context_manager": "🧠 Menganalisis Konteks & Rute Pertanyaan",
    "legal_search": "🔍 Mencari Referensi Pasal Hukum di Database Qdrant",
    "legal_reasoner": "⚖️ Menyusun Analisis & Argumen Hukum",
    "output_guard": "🔬 Memverifikasi Akurasi dan Validitas Jawaban",
}


@cl.on_chat_start
async def start_chat():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    await cl.Message(
        content="Halo! Saya adalah **Asisten Hukum Ketenagakerjaan Indonesia**. Silakan tanyakan masalah ketenagakerjaan Anda."
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    payload = {"message": message.content, "session_id": session_id}

    # 1. Wadah kosong jawaban utama
    msg_placeholder = cl.Message(content="")
    await msg_placeholder.send()

    # 2. Indikator proses berpikir tunggal yang aman dari bug avatar
    thinking_step = cl.Step(name="AnalisisSistem", type="tool")
    thinking_step.output = "🤖 Asisten Hukum sedang menganalisis regulasi..."
    await thinking_step.send()

    final_answer = ""
    citations = []

    try:
        # PANGGIL FACTORY: Client otomatis tahu kapan harus pakai token GCP, kapan tidak.
        async with create_backend_client() as http_client:
            async with http_client.stream(
                "POST", FASTAPI_CHAT_URL, json=payload
            ) as response:
                if response.status_code != 200:
                    if thinking_step:
                        await thinking_step.remove()
                    msg_placeholder.content = f"❌ Server merespons dengan status eror: {response.status_code}"
                    await msg_placeholder.update()
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        json_str = line[6:]
                        data = json.loads(json_str)
                        event_type = data.get("event")

                        # A. Update teks status berpikir berdasarkan rute aktual backend
                        if event_type == "node_complete":
                            completed_node = data.get("node")
                            if thinking_step:
                                if completed_node == "input_guard":
                                    thinking_step.output = "🛡️ Keamanan input tervalidasi. Menganalisis konteks hukum..."
                                elif completed_node == "context_manager":
                                    thinking_step.output = "🧠 Mengekstrak referensi pasal dan memproses database..."
                                elif completed_node == "legal_search":
                                    thinking_step.output = "🔍 Menyusun argumentasi hukum dan pasal ketenagakerjaan..."
                                elif completed_node == "legal_reasoner":
                                    thinking_step.output = "⚖️ Memverifikasi validitas akhir dan akurasi jawaban..."
                                await thinking_step.update()

                        # B. Tangkap hasil akhir utuh yang sudah lolos uji Kelayakan (Output Guard)
                        elif event_type == "final_result":
                            if thinking_step:
                                await thinking_step.remove()
                                thinking_step = None

                            final_answer = data.get("final_answer", "")
                            citations = data.get("citations", [])

                # 3. Pengaman visual jika loader masih tertinggal
                if thinking_step:
                    await thinking_step.remove()

                # 4. Render Dokumen Sitasi ke layar Sidebar jika ada
                elements = []
                for idx, cite in enumerate(citations, start=1):
                    elements.append(
                        cl.Text(
                            name="Daftar Sitasi Kutipan Pasal",
                            content=f"**{cite['uu_id']} Pasal {cite['pasal']} Ayat {cite['ayat']}** \n{cite['text_child']}",
                            display="side",
                        )
                    )

                # 5. Cetak jawaban hukum final utuh ke layar secara instan dan bersih
                msg_placeholder.content = final_answer
                if elements:
                    msg_placeholder.elements = elements
                await msg_placeholder.update()

    except httpx.RequestError as exc:
        if thinking_step:
            await thinking_step.remove()
        msg_placeholder.content = f"⚠️ Gagal terhubung ke Backend API Utama: {exc}"
        await msg_placeholder.update()
