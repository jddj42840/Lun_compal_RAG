import os
import queue
import re
import time
import json
import requests
import gradio as gr
import pandas as pd
from utils.logging_colors import logger
from utils.chat import Chat_api

protocal = os.getenv("PROTOCAL", "http")
url = os.getenv("SERVER_URL", "10.20.1.96")
port = os.getenv("PORT", "5001")

# 用來控制多人同時 Submit 要處理的請求佇列，多個請求傳入時最多只接受一個請求
submit_queue = queue.Queue(maxsize=1)
'''用來控制多人同時 Submit 要處理的請求佇列，多個請求傳入時最多只接受一個請求'''


class LLM:
    def update_model_list() -> list:
        try:
            text_model_list = json.loads(requests.get(f"{protocal}://{url}:{port}/v1/internal/model/list", timeout=(10, None)).text)["model_names"]
            except_file = ["wget-log", "output.log", "Octopus-v2", "Phi-3-mini-4k-instruct", ]
            text_model_list = [f for f in text_model_list if f not in except_file]
            logger.info("Model list fetched successfully")
            return gr.update(choices=text_model_list, value=LLM.get_model())
        except requests.exceptions.ConnectionError:
            logger.error("ConnectionError: Failed to connect to the server.")
            return ["None"]
        except requests.exceptions.Timeout:
            logger.error("Timeout: The request timed out.")
            return ["None"]
        
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

    def send_query(text_dropdown: str, text: str, detail_output_box: list, summary_output_box: list, topk: str, **kwargs):
        if LLM.get_model() == 'None':
            gr.Warning("Please selet a language model")
            return False
        
        if text == '':
            gr.Warning("Please enter a question")
            return False
        
        detail_output_box.append([text, ""])
        summary_output_box.append([text, ""])
        
        csv = pd.read_csv("./standard_response.csv", on_bad_lines='skip')
        if (text in csv["Q"].values):
            detail_output_box[-1][1] = csv[csv["Q"] == text]["A(detail)"].values[0]
            summary_output_box[-1][1] = csv[csv["Q"] == text]["A(summary)"].values[0]
            yield "", detail_output_box, summary_output_box, gr.update(visible=False)
            return True

        if submit_queue.full():
            logger.warning("Queue is full. Please wait a minute to execute send query operation!")
            gr.Warning("Queue is full. Please wait a minute to execute send query operation!")
            return False
        else:
            model_data = json.load(open("config.json", "r", encoding="utf-8"))["model_config"]
            if re.search(r"gguf", text_dropdown):
                args = model_data["gguf"]
                gr.Info("gguf格式的模型可能因為評估題詞而載入過久,請謹慎使用.")
            elif re.search(r"2b|2B|6b|6B|7b|7B|8b|8B|128k", text_dropdown):
                args = model_data["2&7&8B"]
            elif re.search(r"13b|13B", text_dropdown):
                args = model_data["13B"]
            else:
                logger.error(f"Model config not found")
                raise gr.Error("Model config not found")
            logger.info(f"args get")
            
            logger.info("Loading model...")
            try:    
                response = json.loads(requests.post(
                    f"{protocal}://{url}:{port}/v1/internal/model/load",
                    json={
                        "model_name": text_dropdown,
                        "args": args,
                        "settings": {"instruction_template": "Alpaca"}
                    },
                    timeout=(10, None)
                ).text)
                if response["status"] == 0:
                    logger.info(f"model loaded successfully")
                    gr.Info("Model loaded successfully")
                else: 
                    logger.error(f"failed to load model...")
                    raise gr.Error("Failed to load model")
            except requests.exceptions.Timeout:
                logger.error("The request timed out.")
                raise gr.Error("The request timed out.")
            except requests.exceptions.RequestException as e:
                logger.error("Bad Request")
                raise gr.Error("Bad Request")
        
        
        logger.info(f"{text_dropdown} model ready")
        logger.info("add the request into queue...")

        # make response
        chain = Chat_api(temperature=kwargs.get("temperature", 0)).setup_model(search_content=text, topk=topk)
        submit_queue.put(chain)
        start = time.time()
        for chunk in chain.stream("#zh-tw " + text):
            detail_output_box[-1][1] += chunk
            yield "", detail_output_box, summary_output_box, gr.update(visible=False)
        end = time.time()
        logger.info(f"Time cost: {end-start}")
        
        chain = Chat_api(kwargs.get("temperature", 0), custom_instruction=detail_output_box[-1][1]).setup_model(search_content=text, topk=topk)
        start = time.time()
        for chunk in chain.stream("#zh-tw " + text):
            summary_output_box[-1][1] += chunk
            yield "", detail_output_box, summary_output_box, gr.update(visible=False)
        end = time.time()
        logger.info(f"Time cost: {end-start}")
        
        yield "", detail_output_box, summary_output_box, gr.update(visible=True)
        logger.info("remove the request from queue...")
        submit_queue.get()
        
    def load_model(model: str):
        if model == None:
            yield "Please selet a language model"
            gr.Warning("Please selet a language model")
            return False
        
        if submit_queue.full():
            logger.warning("Queue is full. Please wait a minute to execute load model operation.")
            gr.Warning("Queue is full. Please wait a minute to execute load model operation.")
            return False
        else:
            logger.info("Queue is empty.")
            
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
            gr.Warning("Model config not found")
            return False
        
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
                gr.Warning("Failed to load model")
                return False
        except requests.exceptions.Timeout:
            logger.error("The request timed out.")
            yield "Failed to load model."
            gr.Warning("The request timed out.")
            return False
        except requests.exceptions.RequestException as e:
            logger.error("Bad Request")
            yield "Failed to load model."
            gr.Warning("Bad Request")
            return False

    def unload_model(model: str):
        logger.info(f"will unload {model}")
        try:
            requests.post(f"{protocal}://{url}:{port}/v1/internal/model/unload", timeout=(10, None))
            logger.info(f"model unloaded successfully")
            gr.Info("Model unloaded successfully")
            yield "Model unloaded successfully"
        except requests.exceptions.ConnectionError:
            logger.error("ConnectionError: Failed to connect to the server.")
            yield "Failed unload model."
            gr.Warning("Failed to connect to the server.")
            return False
        except requests.exceptions.Timeout:
            logger.error("Timeout: The request timed out.")
            yield "Failed unload model."
            gr.Warning("The request timed out.")
            return False

    def get_model() -> str | bool:
        try:
            response = json.loads(requests.get(f"{protocal}://{url}:{port}/v1/internal/model/info", timeout=(10, None)).text)
            return response["model_name"]
        except requests.exceptions.ConnectionError:
            logger.error("ConnectionError: Failed to connect to the server.")
            gr.Warning("Failed to connect to the server.")
            return False
        except requests.exceptions.Timeout:
            logger.error("Timeout: The request timed out.")
            gr.Warning("The request timed out.")
            return False
