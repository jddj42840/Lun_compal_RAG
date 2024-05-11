import gradio as gr
from utils.llm import LLM
from utils.qdrant import Qdrant
from utils.file_process import File_process

# embedding_model_list = ['BAAI/bge-base-en', 'BAAI/bge-base-en-v1.5', 'BAAI/bge-large-en-v1.5', 'BAAI/bge-small-en', 'BAAI/bge-small-en-v1.5', 'BAAI/bge-small-zh-v1.5', 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'nomic-ai/nomic-embed-text-v1', 'nomic-ai/nomic-embed-text-v1.5', 'thenlper/gte-large', 'mixedbread-ai/mxbai-embed-large-v1', 'intfloat/multilingual-e5-large', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'jinaai/jina-embeddings-v2-base-en', 'jinaai/jina-embeddings-v2-small-en']


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Compal RAG")
    with gr.Row():
        with gr.Column():
            # embed_model_dropdown = gr.Dropdown(label="Embedding Model", choices=embedding_model_list, allow_custom_value=True)
            upload = gr.Files(label="Upload file")
            
            with gr.Row():
                text_model_dropdown = gr.Dropdown(
                    label="Language Model", choices=LLM.get_model_list(), 
                    allow_custom_value=True, value="gemma-7b-it")
            
            with gr.Row():
                load_btn = gr.Button(value="Load model")
                unload_btn = gr.Button(value="Unload model")
            
            msg_box = gr.Textbox(label="輸入問題")
            submit_btn = gr.Button(value="Submit", interactive=True)
            clear_button = gr.ClearButton()
            status = gr.Markdown(value="")
            
        detail_output_box = gr.Chatbot(label="Detail output:", height=800)
        summary_output_box = gr.Chatbot(label="Summary output:", height=800)
    
    submit_btn.click(LLM.send_query, inputs=[text_model_dropdown, msg_box, detail_output_box, summary_output_box], outputs=[msg_box, detail_output_box, summary_output_box])
    load_btn.click(LLM.load_model, inputs=[text_model_dropdown], outputs=[status])
    unload_btn.click(LLM.unload_model, outputs=[status])
    upload.upload(File_process.load_file, inputs=[upload], outputs=[status])
    clear_button.add([detail_output_box, summary_output_box, msg_box])
    
if __name__ == "__main__":
    Qdrant.start_qdrant_db(base_url="ssh://raspi@192.168.1.72")
    demo.launch(server_port=7861, server_name="0.0.0.0")