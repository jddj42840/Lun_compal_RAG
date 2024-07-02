import os
import time
import json
import uuid
import kombu
import redis
import requests
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True, dotenv_path=".env")
from utils.logging_colors import logger
from utils.qdrant import qdrant_client, embedding_model_list
from textwrap import dedent
from fuzzywuzzy import process
from celery import Celery, chain
from celery.exceptions import SoftTimeLimitExceeded
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler, StdOutCallbackHandler

protocal = os.getenv("PROTOCAL", "http")
url = os.getenv("SERVER_URL", "10.20.1.96")
port = os.getenv("PORT", "5001")
celery_app = Celery("tasks", broker_connection_retry_on_startup=True)
celery_app.config_from_object('utils.celery_config')
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=os.getenv("REDIS_PORT", "6379"),
    db=0,
    socket_timeout=10,
    socket_connect_timeout=2
)

# custom prompt 正式版後續要移除


class LLM:
    def get_model_list() -> list:
        try:
            text_model_list = json.loads(requests.get(f"{protocal}://{url}:{port}/v1/internal/model/list",
                                                      timeout=(10, None)).text)["model_names"]
            except_file = ["wget-log", "output.log", "Octopus-v2",
                           "Phi-3-mini-4k-instruct", "bge-reranker-large"]
            text_model_list = [
                f for f in text_model_list if f not in except_file and "gguf" not in f]
            logger.info("Model list fetched successfully")
            return text_model_list
        except requests.exceptions.ConnectionError:
            logger.error("ConnectionError: Failed to connect to the server.")
            return ["None"]
        except requests.exceptions.Timeout:
            logger.error("Timeout: The request timed out.")
            return ["None"]

    def send_query(text_dropdown: str, user_question: str, detail_output_box: list,
                   summary_output_box: list, embed_model: str, topk: str, score_threshold: float,
                   temperature: str = "0", prompt: str = "", **kwargs):
        """
        向 LLM 發送問題請求

        Parameters:
            * text_dropdown (str): 使用者選擇的語言模型
            * user_question (str): 使用者輸入的問題
            * detail_output_box (list): 
            * summary_output_box (list):
            * embed_model (str): 使用者選擇的詞嵌入模型
            * topk (str): 使用者設定的 top-K
            * score_threshold (float): 使用者設定的 score threshold
            * temperature (str): 使用者設定的 temperature
            * prompt (str): 測試設定的 prompt
        """
        if text_dropdown == '':
            gr.Warning("請選擇一語言模型")
            yield "", detail_output_box, summary_output_box, gr.update(), gr.update()
            return False

        if embed_model == None:
            gr.Warning("請選擇一詞嵌入模型")
            yield "", detail_output_box, summary_output_box, gr.update(), gr.update()
            return False
        elif embed_model not in embedding_model_list:
            gr.Warning("無相關詞嵌入模型")
            yield "", detail_output_box, summary_output_box, gr.update(), gr.update()
            return False

        if user_question == '':
            gr.Warning("請輸入問題")
            yield "", detail_output_box, summary_output_box, gr.update(), gr.update()
            return False

        detail_output_box.append([user_question, ""])
        summary_output_box.append([user_question, ""])

        # 判斷是否有匹配到歷史回答紀錄
        if 'match_bool' not in kwargs:
            # dropna 去掉csv中空(nan)的資料欄位
            csv = pd.read_csv("./standard_response.csv", on_bad_lines='skip').dropna(how="any")
            question_list = csv["Q"].values
            if not question_list.size:
                match_score = 0
            else:
                match_question, match_score = process.extractOne(
                    user_question,
                    question_list
                )
                
            if match_score == 100:
                question_index = question_list.tolist().index(match_question)
                detail_output_box[-1][1] = csv["A(detail)"].values[question_index]
                summary_output_box[-1][1] = csv["A(summary)"].values[question_index]
                yield "", detail_output_box, summary_output_box, gr.update(visible=True), gr.update()
                return True
            elif match_score > 0:
                # question_search task
                question_search_task_id = str(uuid.uuid4())
                logger.info(f"Sending question search. task_ID: {question_search_task_id}")
                try:
                    question_search_task.delay(question_search_task_id, question_list.tolist(), temperature, user_question, topk, embed_model, prompt, score_threshold)
                except kombu.exceptions.OperationalError:
                    logger.error("Failed to connect to redis broker.")
                    gr.Warning("連線錯誤.")
                    yield "", detail_output_box, summary_output_box, gr.update(), gr.update()
                    return False
                
                while True:
                    result = redis_client.lrange(question_search_task_id, 0, -1)
                    if result == []:
                        continue
                    if result[-1] == b"#timeout#":
                        logger.error(f"Task {question_search_task_id} exceeded the soft time limit.")
                        gr.Warning("任務超時")
                        yield "", detail_output_box, summary_output_box, gr.update(), gr.update()
                        return False
                    if result[-1] == b"#end#":
                        result = result[0].decode("utf-8").replace("`", "").replace("json", "")
                        logger.info(f"[Question search output (task_ID: {question_search_task_id})]: {result}")
                        break

                if '沒有相關資料' not in result:
                    try:
                        temp = json.loads(result)
                    except json.JSONDecodeError:
                        logger.error(f"JSONDecodeError: Failed to decode the JSON. [Question]: {user_question}")
                        gr.Warning("未預期錯誤...")
                        yield "", detail_output_box, summary_output_box, gr.update(), gr.update()
                        return False
                    match_question, match_score = process.extractOne(
                        temp["question"],
                        question_list
                    )
                    logger.info(f"Match score: {match_score}")
                    if match_score == 100:
                        question_index = question_list.tolist().index(match_question)
                        detail_output_box[-1][1] = csv["A(detail)"].values[question_index]
                        summary_output_box[-1][1] = csv["A(summary)"].values[question_index]
                        yield "", detail_output_box, summary_output_box, gr.update(visible=True), gr.update()
                        return True

        # 建立uuid
        detail_task_id = str(uuid.uuid4())
        summary_task_id = str(uuid.uuid4())

        tasks_chain = chain(
            detail_task.s(
                detail_task_id, 
                temperature, 
                user_question, 
                topk,
                embed_model, 
                prompt, 
                score_threshold
            ),
            summary_task.s(
                summary_task_id, 
                detail_task_id,
                temperature, 
                user_question, 
                topk, embed_model, 
                score_threshold
            )
        )
        
        tasks_chain.apply_async()

        # detail task
        logger.info(f"Sending detail task. task_ID: {detail_task_id}")
        start = time.time()
        while True:
            output = redis_client.lrange(detail_task_id, 0, -1)
            if output == []:
                continue
            if output[-1] == b"#timeout#":
                logger.error(f"Task {detail_task_id} exceeded the soft time limit.")
                gr.Warning("任務超時")
                return False
            if output[-1] == b"#end#":
                break
            output = "".join(index.decode("utf-8") for index in output)
            detail_output_box[-1][1] = output
            yield "", detail_output_box, summary_output_box, gr.update(visible=False), gr.update()
        end = time.time()
        logger.info(
            f"[Detail output (task_ID: {detail_task_id}]: {detail_output_box[-1][1]} ,time cost: {end-start}")

        # summary task
        logger.info(f"Sending summary task. task_ID: {summary_task_id}")
        start = time.time()
        while True:
            output = redis_client.lrange(summary_task_id, 0, -1)
            if output == []:
                continue
            if output[-1] == b"#timeout#":
                logger.error(f"Task {summary_task_id} exceeded the soft time limit.")
                gr.Warning("任務超時")
                return False
            if output[-1] == b"#end#":
                break
            output = "".join(index.decode("utf-8") for index in output)
            summary_output_box[-1][1] = output
            yield "", detail_output_box, summary_output_box, gr.update(visible=False), gr.update()
            time.sleep(0.05)
        end = time.time()
        logger.info(
            f"[Detail output (task_ID: {summary_task_id}]: {detail_output_box[-1][1]} ,time cost: {end-start}")
        yield "", detail_output_box, summary_output_box, gr.update(visible=True), ""

        logger.info("All tasks completed.")

    def get_model() -> str | bool:
        try:
            response = json.loads(requests.get(f"{protocal}://{url}:{port}/v1/internal/model/info",
                                               timeout=(10, None)).text)
            return response["model_name"]
        except requests.exceptions.ConnectionError:
            logger.error("Conne ctionError: Failed to connect to the server.")
            gr.Warning("無法連接至伺服器.")
            return False
        except requests.exceptions.Timeout:
            logger.error("Timeout: The request timed out.")
            gr.Warning("連線逾時.")
            return False


class Chat_api:
    QUESTION_SEARCH_SYS_PROMPT = """以下的參考資料為一系列的問題敘述，參考資料中使用"--------------------------"來區分不同的問題。
若參考資料中沒有與使用者問題敘述最相似的資料，則回答"沒有相關資料"，並停止回答。

若參考資料中有與使用者問題敘述相似的問題，就使用 JSON 格式輸出，KEY 為"question"，VALUE 為<最相近的資料>，並且 VALUE 的內容要與參考資料中的問題完全一致。

輸出必須使用繁體中文。"""

    RAG_DETAIL_SYS_PROMPT = dedent("""你是一個聊天機器人，你必須詳讀參考資料中的不同段落的資訊來回答使用者問題，不要回答沒有在參考資料中的資訊。

若沒有辦法從以下參考資料中取得資訊或參考資料為空白，則回答"沒有相關資料"，則請不要回覆任何參考檔案名稱及頁碼。

輸出必須使用繁體中文，並且在回答的最後加入參考檔案名稱及頁碼。""")

    RAG_SUMMARY_SYS_PROMPT = dedent("""你是一個客服聊天機器人，請將使用者提供的敘述做summary, 回答越精簡越好, 若參考資料為"沒有相關資料"時, 則只回覆"沒有相關資料即可", 若提供的內容中有參考資料, 請在回答中加入參考資料檔案的名稱與頁碼，並且用繁體中文回覆。""")

    def __init__(self, temperature: float = 0, role: str = "assistant", streaming: bool = True):
        """

        Parameters:
        * temperature(str): 前端可以設定，決定模型回覆固定參數
        * role(str): 若使用text-generation-webui 則不須更動，若不是則須查詢role的角色為何
        * custom_content(str): 決定要給模型的參考資料
        * streaming(bool): 是否啟用串流輸出
        """
        
        self.temperature = temperature
        self.role = role
        self.streaming = streaming

    def setup_model(self, score_threshold: float, embed_model: str, user_question: str = "",
                 custom_content: str = "", topk: str = "5", custom_prompt: str = "", 
                 question_type: str = None, **kwargs) -> ChatOpenAI:
        """

        Parameters:
        * score_threshold(int): 相似搜尋閥值
        * embed_model(str): 決定要使用的詞嵌入模型
        * user_question(str): 使用者的提問
        * topk(str): 相似搜尋搜尋的筆數
        * custom_prompt(str): 決定要給系統的題詞
        * question_type(str): 決定當前的請求為語意、詳細、摘要的種類
        """
        if topk == "":
            topk = "5"

        if score_threshold == 0:
            score_threshold = None

        qdrant_client.set_model(embed_model, cache_dir="./.cache")
        result = qdrant_client.query(
            collection_name=embed_model.replace("/", "_"),
            query_text=user_question,
            limit=int(topk),
            score_threshold=score_threshold
        )

        content = "\n\n--------------------------\n\n".join(
            text.metadata["document"] for text in result)


        if custom_prompt != "":
            PROMPT = custom_prompt
        else:
            match question_type:
                case "summary":
                    PROMPT = self.RAG_SUMMARY_SYS_PROMPT
                case "detail":
                    PROMPT = self.RAG_DETAIL_SYS_PROMPT
                case "question_search":
                    PROMPT = self.QUESTION_SEARCH_SYS_PROMPT
        
        prompt_template = f"""{PROMPT}
        
        # 參考資料
        {{content}} 

        # 使用者問題
        Question: {{question}}"""
        
        self.chain = (
            {"content":lambda x:  content if custom_content == "" else custom_content, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(prompt_template)
            | ChatOpenAI(streaming=self.streaming, max_tokens=0, temperature=self.temperature,
                         callbacks=[StreamingStdOutCallbackHandler() if self.streaming else StdOutCallbackHandler()])
            | StrOutputParser()
        )
        return self.chain

@celery_app.task
def question_search_task(question_search_task_id: str, question_list: list, temperature: str, text: str,
                topk: str,  embed_model: str, prompt: str, score_threshold: str):
    try:
        questions = "\n--------------------------\n"
        question_search_api = Chat_api(
            temperature=float(temperature),
            streaming=False
        )
        question_search_chain = question_search_api.setup_model(
            user_question=text, 
            topk=topk, 
            embed_model=embed_model, 
            custom_content=questions.join(question_list), 
            score_threshold=float(score_threshold),
            custom_prompt=prompt, 
            question_type="question_search"
        )
        
        result = question_search_chain.invoke("#zh-tw " + text)
        redis_client.rpush(question_search_task_id, result)
        redis_client.rpush(question_search_task_id, "#end#")
        return True
    except SoftTimeLimitExceeded:
        logger.error(f"Task {question_search_task_id} exceeded the soft time limit.")
        redis_client.rpush(question_search_task_id, "#timeout#")
        return False


@celery_app.task
def detail_task(detail_task_id: str,  temperature: str, text: str,
                topk: str,  embed_model: str, prompt: str, score_threshold: str):
    try:
        detail_api = Chat_api(temperature=float(temperature))
        detail_chain = detail_api.setup_model(
            user_question=text, 
            topk=topk, 
            embed_model=embed_model, 
            custom_prompt=prompt,score_threshold=float(score_threshold),
            question_type="detail"
        )
        
        for chunk in detail_chain.stream("#zh-tw " + text):
            redis_client.rpush(detail_task_id, chunk)
        redis_client.rpush(detail_task_id, "#end#")
        return True
    except SoftTimeLimitExceeded:
        logger.error(f"Task {detail_task_id} exceeded the soft time limit.")
        redis_client.rpush(detail_task_id, "#timeout#")
        return False


@celery_app.task
def summary_task(_, summary_task_id: str, detail_task_id: str, temperature: str, text: str,
                 topk: str, embed_model: str, score_threshold: str):
    try:
        output = redis_client.lrange(detail_task_id, 0, -1)
        # 去除用來判斷模型是否生成完畢的標記("#end#")
        output.pop()

        summary_api = Chat_api(temperature=float(temperature))
        summary_chain = summary_api.setup_model(
            user_question=text, 
            topk=topk, 
            embed_model=embed_model, 
            custom_content="".join(index.decode("utf-8") for index in output),
            score_threshold=float(score_threshold), 
            question_type="summary"
        )
        
        for chunk in summary_chain.stream("#zh-tw " + text):
            redis_client.rpush(summary_task_id, chunk)
        redis_client.rpush(summary_task_id, "#end#")
        return True
    except SoftTimeLimitExceeded:
        logger.error(f"Task {summary_task_id} exceeded the soft time limit.")
        redis_client.rpush(summary_task_id, "#timeout#")
        return False
