# utils/reranker_cross_encoder.py
from typing import List, Tuple
import yaml
from sentence_transformers import CrossEncoder
# CrossEncoder is a model class that takes a pair of sentences (or a query and a document chunk) as 
# input and outputs a score or label indicating their relationship (e.g., relevance, similarity)
from langchain_core.documents import Document #to return document object from the function

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
class CrossEncoderReranker:
    """
    Cross-encoder re-ranker using sentence-transformers.
    Scores (query, chunk) pairs and returns documents sorted by score (desc).
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        model_name = self.config["reranking"]["cross_encoder_model"]
        # trust_remote_code=False is default; model truncates pairs internally
        self.model = CrossEncoder(model_name)
        
        # Optional clip to keep inputs compact (helps speed/cost for long docs)
        #while reranking it will only check the similarity (query, chunk) from first 1200 characters of
        # each chunk and give ranking score based on that. This is to avoid long chunks which may be 
        # costly and slow for reranking. 
        self.max_chars = int(self.config["reranking"].get("max_chunk_chars", 1200))
        
    #thiS function takes a text input and clips it to max_chars if defined above or otherwise returns
    # the text as is. This is useful for long documents.   
    def _clip(self, text: str) -> str:
        if self.max_chars and len(text) > self.max_chars:
            return text[: self.max_chars]
        return text
    
    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Tuple[Document, float]]:
        """
        Returns top_k documents with cross-encoder scores.
        """
        if not docs:
            return []
        
        pairs = [(query, self._clip(d.page_content)) for d in docs] #this creates a list of tuples where
        #each tuple is (query, clipped_doc_content) for each document in docs.
        
        # Use the cross-encoder model to score each (query, document) pair for relevance.
        # Example: For a query and 2 docs, returns scores like [0.95, 0.30] showing how well each 
        # doc answers the query.
        scores = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)

        # Build (doc, score) tuples and sort by score desc
        # Combine each doc with its score and sort them from highest to lowest score (most relevant first).
        # Example: If docs = [doc1, doc2], scores = [0.8, 0.3], this makes [(doc1, 0.8), (doc2, 0.3)]
        # and sorts them as [(doc1, 0.8), (doc2, 0.3)] (highest score first).
        ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]