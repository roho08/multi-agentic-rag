"""Researcher subgraph for query generation and document retrieval."""
import os

from dotenv import load_dotenv

load_dotenv()
from typing import Any, TypedDict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank

from subgraph.graph_states import ResearcherState, QueryState
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from utils.local_llm import chat_model
from utils.utils import config

import logging


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

VECTORSTORE_COLLECTION = config["retriever"]["collection_name"]
VECTORSTORE_DIRECTORY = config["retriever"]["directory"]
TOP_K = config["retriever"]["top_k"]
TOP_K_COMPRESSION = config["retriever"]["top_k_compression"]
ENSEMBLE_WEIGHTS = config["retriever"]["ensemble_weights"]
COHERE_RERANK_MODEL = config["retriever"]["cohere_rerank_model"]


# ============================================================
# Vector Store
# ============================================================

def _setup_vectorstore() -> Chroma:

    embeddings = HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )

    return Chroma(

        collection_name=VECTORSTORE_COLLECTION,

        embedding_function=embeddings,

        persist_directory=VECTORSTORE_DIRECTORY,

    )


def _load_documents(
    vectorstore: Chroma,
) -> list[Document]:
    """Load all documents stored in Chroma."""

    all_data = vectorstore.get(
        include=["documents", "metadatas"]
    )

    documents: list[Document] = []

    for content, meta in zip(
        all_data["documents"],
        all_data["metadatas"],
    ):

        if meta is None:
            meta = {}

        elif not isinstance(meta, dict):
            raise ValueError(
                f"Expected metadata to be a dict, "
                f"but got {type(meta)}"
            )

        documents.append(
            Document(
                page_content=content,
                metadata=meta,
            )
        )

    return documents


# ============================================================
# Retriever Construction
# ============================================================

def _build_retrievers(
    documents: list[Document],
    vectorstore: Chroma,
) -> ContextualCompressionRetriever:
    """
    Build hybrid retrieval pipeline:

    BM25
        +
    Vector similarity
        +
    MMR
        ↓
    Ensemble
        ↓
    Cohere reranking
    """

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    retriever_bm25 = BM25Retriever.from_documents(
        documents,
        search_kwargs={"k": TOP_K},
    )

    # --------------------------------------------------------
    # Vector similarity
    # --------------------------------------------------------

    retriever_vanilla = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # --------------------------------------------------------
    # MMR
    # --------------------------------------------------------

    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K},
    )

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            retriever_vanilla,
            retriever_mmr,
            retriever_bm25,
        ],
        weights=ENSEMBLE_WEIGHTS,
    )

    # --------------------------------------------------------
    # Cohere Reranker
    # --------------------------------------------------------

    compressor = CohereRerank(
    top_n=TOP_K_COMPRESSION,
    model=COHERE_RERANK_MODEL,
    cohere_api_key=os.getenv("CO_API_KEY"),
)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    return compression_retriever


# ============================================================
# Initialize Retrieval Pipeline
# ============================================================

logger.info("Loading Chroma vector store...")

vectorstore = _setup_vectorstore()

logger.info("Loading documents from vector store...")

documents = _load_documents(vectorstore)

logger.info(
    "Loaded %d documents from vector store.",
    len(documents),
)

logger.info("Building hybrid retriever...")

compression_retriever = _build_retrievers(
    documents,
    vectorstore,
)

logger.info("Hybrid retriever ready.")


# ============================================================
# Query Generation
# ============================================================

def generate_queries(
    state: ResearcherState,
    *,
    config: RunnableConfig,
) -> dict[str, list[str]]:
    """
    Generate multiple search queries using the local Qwen model.
    """

    logger.info("--- GENERATE QUERIES ---")

    messages = [
        {
            "role": "system",
            "content": GENERATE_QUERIES_SYSTEM_PROMPT,
        },
        {
            "role": "human",
            "content": state.question,
        },
    ]

    response = chat_model.invoke(messages)

    text = response.content.strip()

    logger.info(
        "Generated query response: %s",
        text,
    )

    # --------------------------------------------------------
    # Temporary parser
    # --------------------------------------------------------
    #
    # We will make this strict JSON in the next step.
    #
    lines = [
        line.strip("- •123456789. ")
        for line in text.splitlines()
        if line.strip()
    ]

    queries = []

    for line in lines:
        if line and len(queries) < 2:
            queries.append(line)

    # Always include original question.
    queries.append(state.question)

    # Remove duplicates while preserving order.
    queries = list(dict.fromkeys(queries))

    logger.info(
        "Queries: %s",
        queries,
    )

    return {
        "queries": queries
    }


# ============================================================
# Retrieval
# ============================================================

def retrieve_and_rerank_documents(
    state: QueryState,
    *,
    config: RunnableConfig,
) -> dict[str, list[Document]]:
    """Retrieve and rerank documents for one query."""

    logger.info("--- RETRIEVING DOCUMENTS ---")

    logger.info(
        "Query: %s",
        state.query,
    )

    response = compression_retriever.invoke(
        state.query
    )

    logger.info(
        "Retrieved %d documents.",
        len(response),
    )

    return {
        "documents": response
    }


# ============================================================
# Parallel Retrieval
# ============================================================

def retrieve_in_parallel(
    state: ResearcherState,
) -> list[Send]:
    """
    Create one retrieval task for every generated query.
    """

    return [
        Send(
            "retrieve_and_rerank_documents",
            QueryState(query=query),
        )
        for query in state.queries
    ]


# ============================================================
# Build Researcher Graph
# ============================================================

builder = StateGraph(
    ResearcherState
)

builder.add_node(
    "generate_queries",
    generate_queries,
)

builder.add_node(
    "retrieve_and_rerank_documents",
    retrieve_and_rerank_documents,
)

builder.add_edge(
    START,
    "generate_queries",
)

builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,
)

builder.add_edge(
    "retrieve_and_rerank_documents",
    END,
)

researcher_graph = builder.compile()