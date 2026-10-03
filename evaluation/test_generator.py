import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevance, faithfulness

from src.agent.nodes.reasoner import legal_reasoner_node
from src.logger import logger

load_dotenv()


def evaluate_generation_subsystem():
    logger.info("Memulai pengujian komponen Generator (Metode Suapi Fakta)...")

    testset_path = "data/testsets/golden_testset.csv"
    df_testset = pd.read_csv(testset_path)
    generated_answers_list = []

    for index, row in df_testset.iterrows():
        # Menyuapi fakta ideal langsung ke dalam State buatan
        mock_state = {
            "user_input": row["user_input"],
            "messages": [],
            "summary_memory": "",
            "search_filters": {},
            "retrieved_documents": [
                {
                    "source": row.get("source_metadata", "unknown"),
                    "content": row["reference_context"],
                }
            ],
        }
        try:
            node_output = legal_reasoner_node(mock_state)
            generated_answers_list.append(node_output.get("final_answer", ""))
        except Exception as e:
            logger.error(f"Error pada node generator baris #{index + 1}: {e!s}")
            generated_answers_list.append("")

    eval_data = {
        "user_input": df_testset["user_input"].tolist(),
        "response": generated_answers_list,
        "retrieved_contexts": [
            [ctx] for ctx in df_testset["reference_context"].tolist()
        ],
    }

    logger.info("Menghitung metrik pembuatan teks dengan Ragas...")
    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
        )
        report = evaluate(
            dataset=eval_data,
            metrics=[faithfulness, answer_relevance],
            llm=evaluator_llm,
        )

        logger.info(
            f"📊 SKOR GENERATOR -> Faithfulness: {report['faithfulness']:.4f} | Relevance: {report['answer_relevance']:.4f}"
        )
    except Exception as e:
        logger.error(f"Gagal mengevaluasi generator: {e!s}")


if __name__ == "__main__":
    evaluate_generation_subsystem()
