import os

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph

from src.logger import logger

load_dotenv()


def generate_evaluation_assets():
    logger.info("Memulai pembuatan dataset evaluasi...")

    try:
        generator_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        )
        generator_embeddings = LangchainEmbeddingsWrapper(
            GoogleGenAIEmbeddings(model="models/text-embedding-004")
        )
        generator = TestsetGenerator.from_langchain(
            llm=generator_llm, embedding=generator_embeddings
        )
    except Exception as e:
        logger.error(f"Gagal inisialisasi API Gemini: {e!s}")
        return

    graph_path = "data/testsets/knowledge_graph.json"
    if os.path.exists(graph_path):
        logger.info("Memuat Knowledge Graph lokal yang sudah ada...")
        kg = KnowledgeGraph.load(graph_path)
    else:
        logger.info("Membuat Knowledge Graph baru...")
        kg = KnowledgeGraph()

    raw_docs_folder = "data/raw_documents/"
    file_uu_list = [f for f in os.listdir(raw_docs_folder) if f.endswith(".md")]
    logger.info(f"Ditemukan {len(file_uu_list)} file markdown untuk diproses.")

    for file_name in file_uu_list:
        try:
            loader = UnstructuredMarkdownLoader(
                os.path.join(raw_docs_folder, file_name)
            )
            docs = loader.load()
            generator.init_knowledge_graph(docs, kg=kg)
            kg.save(graph_path)
            logger.info(f"File {file_name} berhasil dicicil ke dalam graf lokal.")
        except Exception as e:
            logger.error(f"Gagal memproses file {file_name}: {e!s}")

    logger.info("Mulai merumuskan pertanyaan dari Knowledge Graph...")
    run_config = RunConfig(max_workers=1, timeout=90)

    try:
        testset = generator.generate_with_knowledge_graph(
            kg=kg, testset_size=10, raise_exceptions=False, run_config=run_config
        )

        df = testset.to_pandas()
        if "document_metadata" in df.columns:
            df["source_metadata"] = df["document_metadata"].apply(
                lambda x: (
                    x.get("source", "unknown") if isinstance(x, dict) else "unknown"
                )
            )

        df.to_csv("data/testsets/golden_testset.csv", index=False)
        logger.info(
            "✅ Dataset evaluasi sukses disimpan di 'data/testsets/golden_testset.csv'"
        )
    except Exception as e:
        logger.error(f"Proses pembuatan soal gagal: {e!s}")


if __name__ == "__main__":
    generate_evaluation_assets()
