import os
import re
import json
import PyPDF2
import subprocess
import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from utils.qdrant import qdrant_client, embeddings
from utils.llm import LLM
from utils.logging_colors import logger

class File_process:
    def load_file(file_paths: list):
        if not os.path.exists("config.json"):
            with open("config.json", "w") as f:
                json.dump({"uploaded_file": []}, f)
                
        config_info = json.load(open("config.json", "r", encoding="utf-8"))
        
        yield "Uploading..."
        # upload
        for file_path in file_paths:
            full_file_name =  file_path.name.split("/")[-1]
            file_name, file_extension = os.path.splitext(full_file_name)
            if file_name + file_extension in config_info["uploaded_file"]:
                gr.Info(f"File {full_file_name} already uploaded. Ignore it...")
                continue
            
            yield f"Processing {full_file_name}"
            # 處理不同的副檔格式
            if file_extension == '.pdf':
                pass
            elif file_extension == '.ppt' or file_extension == '.pptx':
                gr.Info("Detect ppt or pptx extension. Please wait for converting...")
                try:
                    subprocess.Popen(["libreoffice", "--headless", "--invisible", "--convert-to", "pdf", file_path, "--outdir", "pdf_output"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()
                    file_path = os.path.join("pdf_output", full_file_name.replace(file_extension, ".pdf"))
                except Exception as e:
                    logger.error(f"An error occurred: {str(e)}")
                    gr.Warning(f"An error occurred: {str(e)}")
                    yield f"Something went wrong. Please try again."
                    return False
            elif file_extension == ".xlsx":
                """
                轉換規則
                1. 有文字的背景顏色盡量一致
                2. 格式(靠左對齊之類的)or縮排盡量一致
                """
                gr.Warning(f"Detect {file_extension} extension. \nATTENTION, upload {file_extension} extension file may loss some information and response unexpected information.")
                subprocess.Popen(["libreoffice", "--headless", "--invisible", "--convert-to", "pdf", file_path, "--outdir", "pdf_output"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()
                file_path = os.path.join("pdf_output", full_file_name.replace(file_extension, ".pdf"))
            elif file_extension == ".csv":
                """
                資料格式如下
                Q,A,Reference
                <問題>,<答案>,<參考資料>
                ...
                """
                df = pd.read_csv(file_path, on_bad_lines='skip')
                document_prefix = f"""**********\n[段落資訊]\n檔案名稱:"{file_name}{file_extension}"\n**********\n\n"""
                csv_data = []
                for index, row in df.iterrows():
                    qa_text = ""
                    qa_text += f"Question: {row['Q']}\n"
                    qa_text += f"Answer: {row['A']}\n"
                    if "Reference" in row:
                        qa_text += f"Reference: {row['Reference']}\n\n"
                    csv_data.append(document_prefix + qa_text)
                    
                qdrant_client.add(
                    collection_name="compal_rag",
                    documents=csv_data)
                continue
            else:
                logger.error(f"File {full_file_name} is not support. Ignore it...")
                gr.Warning(f"File {full_file_name} is not support. Ignore it...")
                yield f"File {full_file_name} is not support."
                continue
            
            # 讀取pdf檔案
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for index in range(len(reader.pages)):
                    document_prefix = f"""**********\n[段落資訊]\n檔案名稱:"{file_name}{file_extension}"\n頁碼:{index+1}\n**********\n\n"""
                    text = reader.pages[index].extract_text()
                    encrypt_string = json.loads(open("./config.json", "r", encoding="utf-8").read())["encrypt_string"]
                    if re.search(encrypt_string, text):
                        gr.Warning(f"Detect encrypt string in page {index+1}. Ignore it...")
                        continue
                    qdrant_client.add(
                        collection_name="compal_rag",
                        documents=[document_prefix + reader.pages[index].extract_text()])
                    # qdrant_client.upload_collection(
                    #     collection_name="compal_rag",
                    #     vectors=[embeddings.encode(document_prefix + reader.pages[index].extract_text())],
                    #     ids=[index + 1],
                    #     payload=[{"document": document_prefix + reader.pages[index].extract_text()}]
                    # )
                    
            # config_info["uploaded_file"].append(full_file_name)
            # json.dump(config_info, open("config.json", "w", encoding="utf-8"))
            
        gr.Info("File uploaded successfully")
        yield "File uploaded successfully"

    def save_answer(choice: gr.Button, text_dropdown: str, text: str, detail_output_box: gr.Chatbot, summary_output_box: gr.Chatbot):
        if choice == "Yes":
            csv = pd.read_csv("./standard_response.csv", on_bad_lines='skip')
            data = {
                "Q": detail_output_box[-1][0], 
                "A(detail)": detail_output_box[-1][1],
                "A(summary)": summary_output_box[-1][1]}
            csv = pd.concat([csv, pd.DataFrame(data, index=[0])], ignore_index=True)
            csv.to_csv("./standard_response.csv", index=False)
            yield gr.update(visible=False), "Save answer successfully."
        else:
            # generate new answer
            yield gr.update(visible=False),  "Generate new answer."
            detail_response = []
            summary_response = []
            for _, detail, summary, _ in LLM.send_query(text_dropdown, text, detail_output_box, summary_output_box):
                detail_response = detail
                summary_response = summary
                yield gr.update(), "", detail_response, summary_response, gr.update()
            
            yield gr.update(visible=True), "", detail_response, summary_response, gr.update()