import os
import re
import time
import json
import requests
import docker
import gradio as gr
import subprocess
import PyPDF2
from textwrap import dedent
from openai import OpenAI
from logging_colors import logger
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StdOutCallbackHandler, StreamingStdOutCallbackHandler
from qdrant_client import QdrantClient, models

load_dotenv(override=True, dotenv_path=".env")
protocal = os.getenv("PROTOCAL", "http")
url = os.getenv("SERVER_URL", "10.20.1.96")
port = os.getenv("PORT", "5001")
client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE", f"{protocal}://{url}:{port}/v1"), 
    api_key=os.getenv("OPENAI_API_KEY", "sk-111111111111111111111111111111111111111111111111"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://25.17.79.112:6333"))
qdrant_client.set_model("intfloat/multilingual-e5-large")


class Functional:
    def get_model_list() -> list:
        try:
            text_model_list = json.loads(requests.get(f"{protocal}://{url}:{port}/v1/internal/model/list", timeout=(10, None)).text)["model_names"]
            except_file = ["wget-log", "output.log", "Octopus-v2", "Phi-3-mini-4k-instruct", ]
            text_model_list = [f for f in text_model_list if f not in except_file]
            logger.info("Model list fetched successfully")
            return text_model_list
        except requests.exceptions.ConnectionError:
            logger.error("ConnectionError: Failed to connect to the server.")
            return ["None"]
        except requests.exceptions.Timeout:
            logger.error("Timeout: The request timed out.")
            return ["None"]

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
                gr.Error("Failed to start qdrant container.")
                return False
        else:
            logger.info("found exist qdrant container, starting container...")
            try:
                client.containers.get("qdrant").start()
            except Exception as e:
                logger.error(str(e))
                gr.Error("Failed to start qdrant container.")
                return False
        
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

    def send_query(text_dropdown: str, text: str, output_box: gr.Chatbot):
        
        if Functional.get_model() == 'None':
            gr.Warning("Please selet a language model")
            return False

        if text == '':
            raise gr.Error("Please enter a question")
        
        output_box.append([text, ""])
        chain = chat_api().setup_model(search_content=text)
        start = time.time()
        for chunk in chain.stream("#zh-tw" + text):
            output_box[-1][1] += chunk
            yield "", output_box
        end = time.time()
        logger.info(f"Time cost: {end-start}")

    def load_file(file_paths: list):
        if not os.path.exists("config.json"):
            with open("config.json", "w") as f:
                json.dump({"uploaded_file": []}, f)
                
        config_info = json.load(open("config.json", "r", encoding="utf-8"))
        
        # upload
        yield "Uploading..."
        
        
        for file_path in file_paths:
            full_file_name =  file_path.name.split("/")[-1]
            file_name, file_extension = os.path.splitext(full_file_name)
            if file_name + file_extension in config_info["uploaded_file"]:
                gr.Info(f"File {full_file_name} already uploaded. Ignore it...")
                continue
            
            if file_extension == '.pdf':
                pass
            elif file_extension == '.ppt' or file_extension == '.pptx':
                gr.Info("Detect ppt or pptx extension. Please wait for converting...")
                try:
                    yield "Converting ppt to pdf..."
                    subprocess.Popen(["libreoffice", "--headless", "--invisible", "--convert-to", "pdf", file_path, "--outdir", "ppt_to_pdf"]).wait()
                    file_path = os.path.join("ppt_to_pdf", full_file_name.replace(file_extension, ".pdf"))
                except Exception as e:
                    logger.error(f"An error occurred: {str(e)}")
                    gr.Error(f"An error occurred: {str(e)}")
            else:
                logger.error(f"File {full_file_name} is not support. Ignore it...")
                gr.Error(f"File {full_file_name} is not support. Ignore it...")
                continue
            
            config_info["uploaded_file"].append(full_file_name)
            json.dump(config_info, open("config.json", "w", encoding="utf-8"))
            
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for index in range(len(reader.pages)):
                    document_prefix = f"""**********\n[段落資訊]\n檔案名稱:"{file_name}"\n頁碼:{index+1}\n**********\n\n"""
                    qdrant_client.add(
                        collection_name="compal_rag",
                        documents=[document_prefix + reader.pages[index].extract_text()])
                
        gr.Info("File uploaded successfully")
        yield "File uploaded successfully"

    def load_model(model: str):
        if model == None:
            yield "Please selet a language model"
            raise gr.Error("Please selet a language model")
        
        model_data = json.load(open("config.json", "r", encoding="utf-8"))["model_config"]
        
        if re.search(r"gguf", model):
            args = model_data["gguf"]
        elif re.search(r"2b|2B|6b|6B|7b|7B|8b|8B|128k", model):
            args = model_data["2&7&8B"]
        elif re.search(r"13b|13B", model):
            args = model_data["13B"]
        else:
            logger.error(f"Model config not found")
            raise gr.Error("Model config not found")
        
        yield "Loading model..."
        try:    
            response = json.loads(requests.post(
                f"{protocal}://{url}:{port}/v1/internal/model/load",
                json={
                    "model_name": model,
                    "args": args,
                    "settings": {"instruction_template": "Alpaca"}
                },
                timeout=(10, None)
            ).text)
            if response["status"] == 0:
                logger.info(f"model loaded successfully")
                yield "Model loaded successfully"
            else: 
                logger.error(f"failed to load model...")
                yield "Failed to load model"
                raise gr.Error("Failed to load model")
        except requests.exceptions.Timeout:
            logger.error("The request timed out.")
            yield "Failed to load model."
            raise gr.Error("The request timed out.")
        except requests.exceptions.TooManyRedirects:
            logger.error("Bad Request")
            yield "Failed to load model."
            raise gr.Error("Bad Request")
        except requests.exceptions.RequestException as e:
            logger.error("Bad Request")
            yield "Failed to load model."
            raise gr.Error("Bad Request")

    def unload_model():
        try:
            requests.post(f"{protocal}://{url}:{port}/v1/internal/model/unload", timeout=(10, None))
            logger.info(f"model unloaded successfully")
            gr.Info("Model unloaded successfully")
            yield "Model unloaded successfully"
        except requests.exceptions.ConnectionError:
            logger.error("ConnectionError: Failed to connect to the server.")
            yield "Failed unload model."
            raise gr.Error("Failed to connect to the server.")
        except requests.exceptions.Timeout:
            logger.error("Timeout: The request timed out.")
            yield "Failed unload model."
            raise gr.Error("The request timed out.")
        except Exception as e:
            logger.error(f"request failed: {str(e)}")
            yield "Failed unload model."
            raise gr.Error(f"request failed: {str(e)}")

    def get_model() -> str:
        response = json.loads(requests.get(f"{protocal}://{url}:{port}/v1/internal/model/info", timeout=(10, None)).text)
        return response["model_name"]


class chat_api:
    """
    is_rag: 是否使用RAG
    temperature: 模型感情
    role: 對話角色
    """
    
    RAG_SYS_PROMPT = dedent("""
        你是一個客服聊天機器人，以下提供的資料為連續或近似的資料，你必須參考以下不同段落中的資訊來回答問題，請注意段落中有可能會有多餘的換行，若有遇到則將上下文連貫起來。
        如果你不知道答案，可以說「抱歉，我沒有任何資訊」，不得創造答案。
        嘗試使用正式語言和專有名詞，保留原文格式（如標題、標點符號、列舉清單（如1-1、1-2...）等）。
        輸出必須使用繁體中文，並且在回答的最後說明參考檔案名稱和頁碼。
    """)
    
    def __init__(self, streaming: bool = True, temperature: float = 0.7, role: str = "assistant"):
        self.temperature = temperature
        self.role = role
        self.system_prompt_prefix = self.RAG_SYS_PROMPT
        self.streaming = streaming
        
    def setup_model(self, search_content: str = "", **kwargs) -> ChatOpenAI:
        self.history_data = []
        result = qdrant_client.query(
            collection_name="compal_rag",
            query_text=search_content,
            limit=kwargs.get("top_k", 8))
        
        content = "\n\n--------------------------\n\n".join(text.metadata["document"] for text in result)
        
        prompt_template = f"""{self.system_prompt_prefix}
        
        {{context}} 

        Question: {{question}}"""
        
        self.chain = (
            {"context": lambda x: content, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(prompt_template)
            | ChatOpenAI(streaming=self.streaming, max_tokens=0, callbacks=[StreamingStdOutCallbackHandler() if self.streaming else StdOutCallbackHandler()], temperature=self.temperature)
            | StrOutputParser()
        )
        return self.chain
