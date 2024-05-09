import os
import time
import docker
import gradio as gr
from utils.logging_colors import logger
from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://192.168.1.72:6333"))
qdrant_client.set_model("intfloat/multilingual-e5-large")

class Qdrant:
    # use linux non-root docker desktop as default method
    def start_qdrant_db(base_url: str = f"unix:///home/{os.getlogin()}/.docker/desktop/docker.sock") -> None :
        try:
            client = docker.DockerClient(base_url=base_url)
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
            time.sleep(5)
        
        collection_list = []
        collections = qdrant_client.get_collections()
        for collection in collections:
            for c in list(collection[1]):
                collection_list.append(c.name)

        if "compal_rag" not in collection_list:
            qdrant_client.recreate_collection(
                collection_name="compal_rag", 
                vectors_config=qdrant_client.get_fastembed_vector_params(),
                optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
                quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8)),
                hnsw_config=models.HnswConfigDiff(on_disk=True))
