import os
import time
import docker
import gradio as gr
import pandas as pd
from utils.logging_colors import logger
from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://192.168.1.72:6333"))

# embedding_model_list = ['BAAI/bge-base-en', 'BAAI/bge-base-en-v1.5', 'BAAI/bge-large-en-v1.5', 'BAAI/bge-small-en', 'BAAI/bge-small-en-v1.5', 'BAAI/bge-small-zh-v1.5', 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'nomic-ai/nomic-embed-text-v1', 'nomic-ai/nomic-embed-text-v1.5', 'thenlper/gte-large', 'mixedbread-ai/mxbai-embed-large-v1', 'intfloat/multilingual-e5-large', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'jinaai/jina-embeddings-v2-base-en', 'jinaai/jina-embeddings-v2-small-en']

"""支援中文的"""
embedding_model_list = ['BAAI/bge-small-zh-v1.5', 'intfloat/multilingual-e5-large']

class Qdrant:
    def qdrant_start_db() -> None :
        try:
            client = docker.DockerClient(base_url=os.getenv("DOCKER_SOCKET_URL", f"unix:///home/{os.getlogin()}/.docker/desktop/docker.sock"))
        except Exception as e:
            logger.error("Failed to start Qdrant container, Error: " + str(e))
            gr.Error("無法啟動Qdrant容器.")
            return False
        
        if client.containers.list(all=True, filters={"name": ["qdrant"]}) == []:
            logger.info("container not exist, creating container...")
            try:
                client.containers.run(
                    "qdrant/qdrant:latest",
                    name="qdrant",
                    ports={"6333/tcp": 6333, "6334/tcp": 6334},
                    detach=True,
                    volumes={f"{os.getcwd()}/qdrant_data": {"bind": "/qdrant/storage", "mode": "rw"}}
                )
            except Exception as e:
                logger.error("Failed to start Qdrant container, Error: " + str(e))
                gr.Error("無法啟動Qdrant容器.")
                return False
            
        else:
            logger.info("found exist qdrant container, starting container...")
            client.containers.get("qdrant").start()
            time.sleep(5)
        
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
                    hnsw_config=models.HnswConfigDiff(on_disk=True, m=48, ef_construct=100)
                )
        
        if not os.path.exists("./standard_response.csv"):
            pd.DataFrame(columns=["Q", "A(detail)", "A(summary)"]).to_csv("./standard_response.csv", index=False)
