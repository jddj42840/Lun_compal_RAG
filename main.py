import gradio as gr
import utils
from logging_colors import logger

# embedding_model_list = ['BAAI/bge-base-en', 'BAAI/bge-base-en-v1.5', 'BAAI/bge-large-en-v1.5', 'BAAI/bge-small-en', 'BAAI/bge-small-en-v1.5', 'BAAI/bge-small-zh-v1.5', 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'nomic-ai/nomic-embed-text-v1', 'nomic-ai/nomic-embed-text-v1.5', 'thenlper/gte-large', 'mixedbread-ai/mxbai-embed-large-v1', 'intfloat/multilingual-e5-large', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'jinaai/jina-embeddings-v2-base-en', 'jinaai/jina-embeddings-v2-small-en']


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Compal RAG")
    with gr.Row():
        with gr.Column():
            # embed_model_dropdown = gr.Dropdown(label="Embedding Model", choices=embedding_model_list, allow_custom_value=True)
            upload = gr.Files(label="Upload file")
            
            with gr.Row():
                text_model_dropdown = gr.Dropdown(label="Language Model", choices=utils.Functional.get_model_list(), allow_custom_value=True)
                refresh_btn = gr.Button(value="", icon="./icons/refresh.png")
                refresh_btn.click(utils.Functional.get_model_list, outputs=[text_model_dropdown])
            
            with gr.Row():
                load_btn = gr.Button(value="Load model")
                unload_btn = gr.Button(value="Unload model")
            
            msg_box = gr.Textbox(label="輸入問題")
            submit_btn = gr.Button(value="Submit", interactive=True)
            clear_button = gr.ClearButton()
            status = gr.Markdown(value="")
            
        output_box = gr.Chatbot(label="Output:", height=800)
    
    submit_btn.click(utils.Functional.send_query, inputs=[text_model_dropdown, msg_box, output_box], outputs=[msg_box, output_box])
    load_btn.click(utils.Functional.load_model, inputs=[text_model_dropdown], outputs=[status])
    unload_btn.click(utils.Functional.unload_model, outputs=[status])
    upload.upload(utils.Functional.load_file, inputs=[upload], outputs=[status])
    clear_button.add([output_box, msg_box])
    
if __name__ == "__main__":
    utils.Functional.start_qdrant_db(base_url="ssh://raspi@25.17.79.112")
    demo.launch(server_port=7861)