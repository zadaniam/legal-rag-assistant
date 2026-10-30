# ui/app_chainlit.py

import json
import uuid

import chainlit as cl
import httpx

from src.logger import logger
from src.utils.gcp_auth import FASTAPI_CHAT_URL, create_backend_client

# Mapping nama Node LangGraph ke Bahasa Indonesia agar tampilan UI profesional
NODE_DISPLAY_MAP = {
    "input_guard": "🛡️ Memeriksa Keamanan Input Pengguna",
    "context_manager": "🧠 Menganalisis Konteks & Rute Pertanyaan",
    "legal_search": "🔍 Mencari Referensi Pasal Hukum di Database Qdrant",
    "legal_reasoner": "⚖️ Menyusun Analisis & Argumen Hukum",
    "output_guard": "🔬 Memverifikasi Akurasi dan Validitas Jawaban",
}

# Mapping nama database UU ke format penulisan resmi Indonesia
UU_DISPLAY_MAP = {
    "uu_ketenagakerjaan_2003": "UU No. 13 Tahun 2003 tentang Ketenagakerjaan",
    "uu_bpjs_2011": "UU No. 24 Tahun 2011 tentang BPJS",
}


@cl.on_chat_start
async def start_chat():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    # Log awal sesi baru dengan context session_id
    logger.info(f"[{session_id}] Sesi chat baru dimulai oleh pengguna.")

    await cl.Message(
        content="Halo! Saya adalah **Asisten Hukum Ketenagakerjaan Indonesia**. Silakan tanyakan masalah ketenagakerjaan Anda."
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    payload = {"message": message.content, "session_id": session_id}

    # Log ketika pesan masuk
    logger.info(f"[{session_id}] Menerima pesan pengguna: {message.content[:50]}...")
    logger.debug(f"[{session_id}] Payload dikirim ke backend: {json.dumps(payload)}")

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
            logger.info(
                f"[{session_id}] Membuka koneksi stream ke Backend API Utama..."
            )

            async with http_client.stream(
                "POST", FASTAPI_CHAT_URL, json=payload
            ) as response:
                if response.status_code != 200:
                    logger.error(
                        f"[{session_id}] Backend merespons gagal dengan status: {response.status_code}"
                    )
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
                            # Log perpindahan status node di LangGraph
                            logger.info(
                                f"[{session_id}] Node selesai dieksekusi: {completed_node}"
                            )

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
                            logger.info(
                                f"[{session_id}] Menerima data final_result dari backend."
                            )
                            if thinking_step:
                                await thinking_step.remove()
                                thinking_step = None

                            final_answer = data.get("final_answer", "")
                            citations = data.get("citations", [])

                            # 🔍 TAMBAHKAN DUA BARIS INSPEKSI INI:
                            logger.debug(
                                f"[{session_id}] ISI REAL FINAL ANSWER: '{final_answer}'"
                            )
                            logger.debug(
                                f"[{session_id}] ISI REAL CITATIONS: {citations}"
                            )

                # 3. Pengaman visual jika loader masih tertinggal
                if thinking_step:
                    await thinking_step.remove()

                # 4. Render Dokumen Sitasi ke layar Sidebar jika ada
                elements = []
                logger.info(
                    f"[{session_id}] Memproses {len(citations)} sitasi hukum untuk ditampilkan."
                )

                for _idx, cite in enumerate(citations, start=1):
                    raw_uu_id = cite.get("uu_id", "")
                    formatted_uu_id = UU_DISPLAY_MAP.get(raw_uu_id, raw_uu_id)

                    markdown_content = (
                        f"## ⚖️ {formatted_uu_id}\n"
                        f"**{cite['bab']}**\n\n"
                        f"{cite['text_parent']}\n\n"
                        f"---\n\n"
                    )

                    elements.append(
                        cl.Text(
                            name="Daftar Sitasi Pasal Terkait",
                            content=markdown_content,
                            display="side",
                        )
                    )

                # 5. Cetak jawaban hukum final utuh ke layar secara instan dan bersih
                msg_placeholder.content = final_answer
                if elements:
                    msg_placeholder.elements = elements
                await msg_placeholder.update()

                logger.info(
                    f"[{session_id}] Jawaban hukum berhasil dirender ke UI pengguna."
                )

    except httpx.RequestError as exc:
        # Log detail error koneksi ke server backend
        logger.exception(
            f"[{session_id}] Terjadi kesalahan jaringan saat menghubungi API: {exc}"
        )

        if thinking_step:
            await thinking_step.remove()
        msg_placeholder.content = f"⚠️ Gagal terhubung ke Backend API Utama: {exc}"
        await msg_placeholder.update()
