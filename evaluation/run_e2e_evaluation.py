import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevance, context_recall, faithfulness

from src.agent.graph import legal_agent_app as app
from src.logger import logger

load_dotenv()


def execute_end_to_end_assessment():
    logger.info("Memulai pengujian aplikasi penuh dari Ujung ke Ujung (End-to-End)...")

    testset_path = "data/testsets/golden_testset.csv"
    df_testset = pd.read_csv(testset_path)

    final_responses = []
    final_contexts = []
    safety_flags_triggered = 0

    for index, row in df_testset.iterrows():
        initial_state = {"messages": [HumanMessage(content=row["user_input"])]}
        try:
            # Menjalankan seluruh alur graf tanpa interupsi
            final_state = app.invoke(initial_state)

            status = final_state.get("system_status", "clear")
            if status != "clear":
                safety_flags_triggered += 1

            final_responses.append(final_state.get("final_answer", ""))

            live_docs = final_state.get("retrieved_documents", [])
            extracted_passages = [
                doc["content"] for doc in live_docs if "content" in doc
            ]
            final_contexts.append(extracted_passages)
        except Exception as e:
            logger.error(f"Error eksekusi graf pada baris #{index + 1}: {e!s}")
            final_responses.append("")
            final_contexts.append([])

    eval_data = {
        "user_input": df_testset["user_input"].tolist(),
        "response": final_responses,
        "retrieved_contexts": final_contexts,
        "reference_context": [
            [ctx] for ctx in df_testset["reference_context"].tolist()
        ],
    }

    logger.info(
        f"Uji coba selesai. Total pelanggaran Guardrails: {safety_flags_triggered}. Menghitung skor akhir..."
    )
    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        )
        report = evaluate(
            dataset=eval_data,
            metrics=[faithfulness, answer_relevance, context_recall],
            llm=evaluator_llm,
        )

        logger.info("=== HASIL EVALUASI AKHIR SISTEM (E2E) ===")
        logger.info(f"Faithfulness     : {report['faithfulness']:.4f}")
        logger.info(f"Answer Relevance : {report['answer_relevance']:.4f}")
        logger.info(f"Context Recall   : {report['context_recall']:.4f}")
        logger.info(f"Guardrail Flags  : {safety_flags_triggered} kejadian")
    except Exception as e:
        logger.error(f"Gagal menjalankan evaluasi E2E: {e!s}")


if __name__ == "__main__":
    execute_end_to_end_assessment()
