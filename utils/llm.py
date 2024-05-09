import os
import re
import time
import json
import requests
import gradio as gr
from utils.logging_colors import logger
from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path=".env")
protocal = os.getenv("PROTOCAL", "http")
url = os.getenv("SERVER_URL", "10.20.1.96")
port = os.getenv("PORT", "5001")


class LLM:
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

    def send_query(text_dropdown: str, text: str, output_box: gr.Chatbot):
        
        if LLM.get_model() == 'None':
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

    def load_model(model: str):
        if model == None:
            yield "Please selet a language model"
            raise gr.Error("Please selet a language model")
        
        model_data = json.load(open("config.json", "r", encoding="utf-8"))["model_config"]
        
        if re.search(r"gguf", model):
            args = model_data["gguf"]
            gr.Info("gguf格式的模型可能因為評估題詞而載入過久,請謹慎使用.")
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


