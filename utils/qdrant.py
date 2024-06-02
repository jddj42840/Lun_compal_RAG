import os
import time
import json
import docker
import gradio as gr
from utils.logging_colors import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://192.168.1.72:6333"))
qdrant_embed_model = "intfloat/multilingual-e5-large"
qdrant_client.set_model(qdrant_embed_model, cache_dir="./.cache")
# qdrant_client.set_model("thenlper/gte-large")

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                                    model_kwargs={'device': 'cpu'}, 
                                    encode_kwargs={'device': 'cpu'})

# embeddings = SentenceTransformer('thenlper/gte-large-zh', device="cpu")
# embeddings = SentenceTransformer('intfloat/multilingual-e5-large', device="cpu")

class Qdrant:
    def start_qdrant_db() -> None :
        try:
            client = docker.DockerClient(base_url=os.getenv("DOCKER_SOCKET_URL", f"unix:///home/{os.getlogin()}/.docker/desktop/docker.sock"))
        except Exception as e:
            logger.error(str(e))
            gr.Error("Failed to start qdrant container.")
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
                logger.error(str(e))
                gr.Error("Failed to create qdrant container.")
                return False
            
        else:
            logger.info("found exist qdrant container, starting container...")
            client.containers.get("qdrant").start()
            time.sleep(3)
        
        collection_list = []
        collections = qdrant_client.get_collections()
        for collection in collections:
            for c in list(collection[1]):
                collection_list.append(c.name)

        print(qdrant_client.get_fastembed_vector_params())
        params = {}
        if "compal_rag" not in collection_list:
            qdrant_client.recreate_collection(
                collection_name="compal_rag", 
                # vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                vectors_config=qdrant_client.get_fastembed_vector_params(),
                optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000))

    def load_embed_model(embed_model: str) -> None:
        if embed_model != qdrant_embed_model:
            gr.Info(f"using {embed_model}. recreating database...")
            qdrant_client.set_model(embed_model, cache_dir="./.cache")
            qdrant_client.recreate_collection(
                collection_name="compal_rag",
                vectors_config=qdrant_client.get_fastembed_vector_params(),
                optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000))
            
            config_info = json.load(open("config.json", "r", encoding="utf-8"))
            config_info["uploaded_file"] = []
            json.dump(config_info, open("config.json", "w", encoding="utf-8"))
            
            gr.Info("Loading complete. Please reupload the document.")
        
        qdrant_client.set_model(embed_model, cache_dir="./.cache")