# utils/two_stage_retriever.py

import yaml
from typing import List
from langchain_core.documents import Document

from utils.retriever_faiss import get_retriever as get_faiss_retriever
from utils.reranker_cross_encoder import CrossEncoderReranker


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TwoStageRetriever:
    """
    Two-Stage Retrieval Pipeline for RAG (Retrieval-Augmented Generation) and will retrieve documents:
    Stage 1: FAISS retrieval (semantic)
    Stage 2: Cross-Encoder reranking (pairwise scoring)
    """

    def __init__(self, config_path: str = "config.yaml", chunks_if_needed=None):
        self.config = load_config(config_path)

        # Stage 1: FAISS retriever (provider-aware)
        self.faiss_retriever = get_faiss_retriever(
            self.config,
            chunks_if_needed=chunks_if_needed
        )

        # Stage 2: Cross-Encoder reranker
        self.reranker = CrossEncoderReranker(config_path)

        # Config params
        self.initial_k = int(self.config["reranking"]["initial_k"]) 
        #for stage-1 retrieval, we will retrieve top initial_k documents from FAISS index
        self.final_k = int(self.config["reranking"]["final_k"])
        #for stage-2 reranking, we will return top final_k documents after reranking the 
        #candidates retrieved in stage-1

    def stage1_retrieve(self, query: str) -> List[Document]:
        """Retrieve large candidate pool (e.g., top 30)."""
        return self.faiss_retriever.invoke(query)[: self.initial_k]

    def stage2_rerank(self, query: str, candidates: List[Document]) -> List[Document]:
        """Apply cross-encoder reranking."""
        ranked = self.reranker.rerank(query, candidates, top_k=self.final_k)
        return [doc for doc, score in ranked]

    def invoke(self, query: str) -> List[Document]:
        """
        Full two-stage pipeline:
        Query → Stage1 FAISS → Stage2 Reranking → Final docs
        """
        # Stage 1
        candidates = self.stage1_retrieve(query)

        if not candidates:
            return []

        # Stage 2
        final_docs = self.stage2_rerank(query, candidates)
        return final_docs