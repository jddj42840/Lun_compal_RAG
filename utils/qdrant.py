import os
import sys
import requests
import pandas as pd
from utils.logging_colors import logger
from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://192.168.1.72:6333"))

# embedding_model_list = ['BAAI/bge-base-en', 'BAAI/bge-base-en-v1.5', 'BAAI/bge-large-en-v1.5', 'BAAI/bge-small-en', 'BAAI/bge-small-en-v1.5', 'BAAI/bge-small-zh-v1.5', 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'nomic-ai/nomic-embed-text-v1', 'nomic-ai/nomic-embed-text-v1.5', 'thenlper/gte-large', 'mixedbread-ai/mxbai-embed-large-v1', 'intfloat/multilingual-e5-large', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'jinaai/jina-embeddings-v2-base-en', 'jinaai/jina-embeddings-v2-small-en']

"""支援中文的"""
embedding_model_list = ['BAAI/bge-small-zh-v1.5', 'intfloat/multilingual-e5-large']

class Qdrant:
    def qdrant_start_db() -> None :
        # check qdrant connection
        try:
            requests.get(os.getenv("QDRANT_URL", "http://192.168.1.72:6333"),
                    timeout=(10, None))
            logger.info("Qdrant connect success.")
        except requests.exceptions.ConnectionError:
            logger.error("Qdrant connect refuse.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Qdrant connection failed. Reason: {str(e)}")
            sys.exit(1)
        
        collection_list = []
        collections = qdrant_client.get_collections()
        for collection in collections:
            for c in list(collection[1]):
                collection_list.append(c.name)
        
        for collection_name in embedding_model_list:
            if collection_name.replace("/", "_") not in collection_list:
                qdrant_client.set_model(collection_name, cache_dir="./.cache")
                qdrant_client.create_collection(
                    collection_name=collection_name.replace("/", "_"),
                    vectors_config=qdrant_client.get_fastembed_vector_params(),
                    optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
                    hnsw_config=models.HnswConfigDiff(on_disk=True, m=64)
                )
        
        if not os.path.exists("./standard_response.csv"):
            pd.DataFrame(columns=["Q", "A(detail)", "A(summary)"]).to_csv("./standard_response.csv", index=False)
