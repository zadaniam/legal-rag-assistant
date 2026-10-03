import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall

from src.agent.graph import legal_agent_app as app
from src.logger import logger

load_dotenv()


def evaluate_retrieval_subsystem():
    logger.info("Memulai pengujian komponen Retriever...")

    testset_path = "data/testsets/golden_testset.csv"
    df_testset = pd.read_csv(testset_path)
    retrieved_contexts_list = []

    for index, row in df_testset.iterrows():
        inputs = {"messages": [HumanMessage(content=row["user_input"])]}
        try:
            # Berhenti tepat setelah node_retriever selesai
            final_state = app.invoke(inputs, interrupt_after=["node_retriever"])
            docs_dict_list = final_state.get("retrieved_documents", [])
            text_pasal_saja = [
                doc["content"] for doc in docs_dict_list if "content" in doc
            ]
            retrieved_contexts_list.append(text_pasal_saja)
        except Exception as e:
            logger.error(f"Error pada baris data #{index + 1}: {e!s}")
            retrieved_contexts_list.append([])

    eval_data = {
        "user_input": df_testset["user_input"].tolist(),
        "retrieved_contexts": retrieved_contexts_list,
        "reference_context": [
            [ctx] for ctx in df_testset["reference_context"].tolist()
        ],
    }

    logger.info("Menghitung metrik pencarian dengan Ragas...")
    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
        )
        report = evaluate(
            dataset=eval_data,
            metrics=[context_recall, context_precision],
            llm=evaluator_llm,
        )

        logger.info(
            f"📊 SKOR RETRIEVER -> Recall: {report['context_recall']:.4f} | Precision: {report['context_precision']:.4f}"
        )
    except Exception as e:
        logger.error(f"Gagal menghitung skor Ragas: {e!s}")


if __name__ == "__main__":
    evaluate_retrieval_subsystem()
