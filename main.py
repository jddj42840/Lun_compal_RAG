import gradio as gr
from dotenv import load_dotenv
load_dotenv(override=True, dotenv_path=".env")
from utils.llm import LLM
from utils.qdrant import Qdrant, embedding_model_list
from utils.file_process import File_process

def advanced_checkbox_change(advanced_checkbox):
    yield gr.update(visible=True) if advanced_checkbox else gr.update(visible=False)

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
                    label="Embedding Model", choices=embedding_model_list, value="intfloat/multilingual-e5-large", 
                    interactive=True, info="詞嵌入模型，主要影響相似度搜尋準確性。") 
                text_model_dropdown = gr.Dropdown(
                    label="Language Model", choices=LLM.get_model_list(), value=lambda: LLM.get_model(), 
                    interactive=True, info="語言模型，負責將相似度搜尋結果資訊做彙整並且輸出成詳細+摘要段落")
                topk = gr.Textbox(label="Top-K", value="5", info="搜尋指定筆數給語言模型做資料彙整")
                score_threshold = gr.Textbox(label="Score Threshold", value="0", visible=True, 
                                             info="相似度門檻閥值，超過閥值的資料才會被列入搜尋結果。multilingual-e5-large建議閥值為0.851, bge-small-zh-v1.5建議閥值為0.51, all-MiniLM-L6-v2建議閥值為0.75, paraphrase-multilingual-MiniLM-L12-v2建議閥值為0.51")
            status = gr.Markdown(value="")
            
        detail_output_box = gr.Chatbot(label="Detail output:", height=800)
        summary_output_box = gr.Chatbot(label="Summary output:", height=800)
    
    submit_btn.click(LLM.send_query, inputs=[text_model_dropdown, msg_box, detail_output_box, summary_output_box, embed_model_dropdown, topk, score_threshold], outputs=[msg_box, detail_output_box, summary_output_box, resp_row])
    upload.upload(File_process.load_file, inputs=[upload], outputs=[status])
    clear_button.add([detail_output_box, summary_output_box, msg_box])
    yes_btn.click(File_process.save_answer, inputs=[yes_btn, text_model_dropdown, detail_output_box, summary_output_box], outputs=[resp_row, status])
    no_btn.click(File_process.save_answer, inputs=[no_btn, text_model_dropdown, detail_output_box, summary_output_box], outputs=[resp_row, msg_box, detail_output_box, summary_output_box, status])
    advanced_checkbox.change(advanced_checkbox_change, inputs=[advanced_checkbox], outputs=[advanced_block])


if __name__ == "__main__":
    Qdrant.start_qdrant_db()
    demo.launch(server_port=7861, server_name="0.0.0.0")