from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_rag"
    embed_model: str = "BAAI/bge-m3"
    generation_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 6
    # Reranking: retrieve rerank_candidates parents, rerank, keep top_k.
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30
    # Abstain (answer "not in the documents") when the best reranked passage
    # scores below this. Chosen from the eval: eval/results/summary.json.
    abstain_threshold: float = 0.05
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
