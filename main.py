import gradio as gr
from dotenv import load_dotenv
load_dotenv(override=True, dotenv_path=".env")
from utils.llm import LLM
from utils.qdrant import Qdrant
from utils.file_process import File_process

# embedding_model_list = ['BAAI/bge-base-en', 'BAAI/bge-base-en-v1.5', 'BAAI/bge-large-en-v1.5', 'BAAI/bge-small-en', 'BAAI/bge-small-en-v1.5', 'BAAI/bge-small-zh-v1.5', 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'nomic-ai/nomic-embed-text-v1', 'nomic-ai/nomic-embed-text-v1.5', 'thenlper/gte-large', 'mixedbread-ai/mxbai-embed-large-v1', 'intfloat/multilingual-e5-large', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'jinaai/jina-embeddings-v2-base-en', 'jinaai/jina-embeddings-v2-small-en']

"""支援中文的"""
embedding_model_list = ['BAAI/bge-small-zh-v1.5', 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'intfloat/multilingual-e5-large']

def advanced_checkbox_change(advanced_checkbox):
    if advanced_checkbox:
        yield gr.update(visible=True)
    else:
        yield gr.update(visible=False)

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Compal RAG")
    with gr.Row():
        with gr.Column():
            with gr.Column(visible=False) as resp_row:
                gr.Markdown("### Is the answer correct?")
                with gr.Row():
                    yes_btn = gr.Button(value="Yes")
                    no_btn = gr.Button(value="No")
                        
            upload = gr.Files(label="Upload file")
            msg_box = gr.Textbox(label="輸入問題")
            advanced_checkbox = gr.Checkbox(label="Advanced Options", value=False)
            submit_btn = gr.Button(value="Submit", interactive=True)
            clear_button = gr.ClearButton()
        
            with gr.Column(visible=False) as advanced_block:
                embed_model_dropdown = gr.Dropdown(
                    label="Embedding Model", choices=embedding_model_list, 
                    interactive=True, info="詞嵌入模型，主要影響相似度搜尋準確性。注意：若選取不同的詞嵌入模型，會重新建立資料庫，以前上傳過的文件須再重新上傳") 
                text_model_dropdown = gr.Dropdown(
                    label="Language Model", choices=LLM.get_model_list(), value=lambda: LLM.get_model(), 
                    interactive=True, info="語言模型，負責將相似度搜尋結果資訊做彙整並且輸出成詳細+摘要段落")
                topk = gr.Textbox(label="Top-K", value="5", info="搜尋指定筆數給語言模型做資料彙整")
            status = gr.Markdown(value="")
            
        detail_output_box = gr.Chatbot(label="Detail output:", height=800)
        summary_output_box = gr.Chatbot(label="Summary output:", height=800)
    
    submit_btn.click(LLM.send_query, inputs=[text_model_dropdown, msg_box, detail_output_box, summary_output_box, topk], outputs=[msg_box, detail_output_box, summary_output_box, resp_row])
    upload.upload(File_process.load_file, inputs=[upload], outputs=[status])
    clear_button.add([detail_output_box, summary_output_box, msg_box])
    yes_btn.click(File_process.save_answer, inputs=[yes_btn, text_model_dropdown, msg_box, detail_output_box, summary_output_box], outputs=[resp_row, status])
    no_btn.click(File_process.save_answer, inputs=[no_btn, text_model_dropdown, msg_box, detail_output_box, summary_output_box], outputs=[resp_row, msg_box, detail_output_box, summary_output_box, status])
    embed_model_dropdown.change(Qdrant.load_embed_model, inputs=[embed_model_dropdown], outputs=[status])
    advanced_checkbox.change(advanced_checkbox_change, inputs=[advanced_checkbox], outputs=[advanced_block])


if __name__ == "__main__":
    Qdrant.start_qdrant_db()
    demo.launch(server_port=7861, server_name="0.0.0.0")