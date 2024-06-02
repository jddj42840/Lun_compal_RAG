from textwrap import dedent
from utils.qdrant import qdrant_client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StdOutCallbackHandler, StreamingStdOutCallbackHandler


class Chat_api:
    """
    is_rag: 是否使用RAG
    temperature: 模型感情
    role: 對話角色
    """
    
    RAG_DETAIL_SYS_PROMPT = dedent("""
        你是一個客服聊天機器人，以下提供的資料為連續或近似的資料，你必須參考以下不同段落中的資訊來回答問題，請注意段落中有可能會有多餘的換行，若有遇到則將上下文連貫起來。
        若問題的答案無法從參考的段落中取得，那你可以參考參考資料來回答答案但是在回覆的開頭必須表明"雖然參考資料中沒有明確的答案，但根據資料....."。
        在最後的回答中，可以衍生出新的問題，但是不得創造新的資訊。
        輸出必須使用繁體中文，並且在回答的最後加入參考檔案名稱及頁碼。
        """)
    
    RAG_SUMMARY_SYS_PROMPT = dedent("""
        你是一個客服聊天機器人，請將使用者提供的敘述做summary, 回答越精簡越好, 若提供的內容中有參考資料, 請在回答中加入參考資料檔案的名稱與頁碼，並且用繁體中文回覆。""")
    
    def __init__(self, streaming: bool = True, temperature: float = 0, role: str = "assistant", custom_instruction: str = ""):
        self.temperature = temperature
        self.role = role
        self.custom_instruction = custom_instruction
        self.streaming = streaming
        
    def setup_model(self, search_content: str = "", topk: str = "5", **kwargs) -> ChatOpenAI:
        if topk == "": 
            topk = "5"
        result = qdrant_client.query(
            collection_name="compal_rag",
            query_text=search_content,
            limit=int(topk))
        
        content = "\n\n--------------------------\n\n".join(text.metadata["document"] for text in result)

        # debug use
        # print(content)
            
        prompt_template = f"""{self.RAG_SUMMARY_SYS_PROMPT if self.custom_instruction != "" else self.RAG_DETAIL_SYS_PROMPT }
        
        {{context}} 

        Question: {{question}}"""
        
        self.chain = (
            {"context": lambda x: content if self.custom_instruction == "" else self.custom_instruction , "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(prompt_template)
            | ChatOpenAI(streaming=self.streaming, max_tokens=0, callbacks=[StreamingStdOutCallbackHandler() if self.streaming else StdOutCallbackHandler()], temperature=self.temperature)
            | StrOutputParser()
        )
        return self.chain
