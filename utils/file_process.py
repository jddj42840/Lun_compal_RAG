import os
import re
import json
import PyPDF2
import subprocess
import gradio as gr
import pandas as pd
from qdrant_client import models
from utils.llm import LLM
from utils.logging_colors import logger
from utils.qdrant import qdrant_client, embedding_model_list


class File_process:
    def load_file(file_paths: list):
        if not os.path.exists("config.json"):
            with open("config.json", "w") as f:
                json.dump({"uploaded_file": []}, f)

        config_info = json.load(open("config.json", "r", encoding="utf-8"))

        yield "Uploading..."
        # upload
        for file_path in file_paths:
            full_file_name = file_path.name.split("/")[-1]
            file_name, file_extension = os.path.splitext(full_file_name)
            if file_name + file_extension in config_info["uploaded_file"]:
                gr.Info(f"{full_file_name} 已經上傳過了，略過此檔案。")
                continue

            yield f"處理 {full_file_name} 中"
            
            # 處理不同的副檔格式
            if file_extension == '.pdf':
                pass
            
            elif file_extension == '.ppt' or file_extension == '.pptx':
                gr.Info(
                    "Detect ppt or pptx extension. Please wait for converting...")
                try:
                    subprocess.Popen(["libreoffice", "--headless", "--invisible", "--convert-to", "pdf", file_path,
                                     "--outdir", "pdf_output"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()
                    file_path = os.path.join(
                        "pdf_output", full_file_name.replace(file_extension, ".pdf"))
                except Exception as e:
                    logger.error(f"An error occurred: {str(e)}")
                    gr.Warning(f"錯誤: {str(e)}")
                    yield f"異常錯誤, 請檢查檔案完整性。"
                    return False
                
            elif file_extension == ".xlsx":
                """
                轉換規則
                1. 有文字的背景顏色盡量一致
                2. 格式(靠左對齊之類的)or縮排盡量一致
                """
                gr.Warning(
                    f"Detect {file_extension} extension. \nATTENTION, upload {file_extension} extension file may loss some information and response unexpected information.")
                subprocess.Popen(
                    ["libreoffice", "--headless", "--invisible", "--convert-to", "pdf", 
                    file_path,
                    "--outdir", "pdf_output"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                ).wait()
                file_path = os.path.join(
                    "pdf_output", 
                    full_file_name.replace(file_extension, ".pdf")
                )
                
            elif file_extension == ".csv":
                """
                // A(summary) 可有可無
                資料格式如下 
                Q,A(detail),A(summary) 
                <問題>,<詳細答案>,<摘要答案>
                ...
                """
                upload_df = pd.read_csv(file_path, on_bad_lines='skip')
                document_prefix = f"""**********\n[段落資訊]\n檔案名稱:"{file_name}{file_extension}"\n**********\n\n"""
                
                answer_csv = pd.read_csv("./standard_response.csv", on_bad_lines='skip')
                for index, row in upload_df.iterrows():
                    if "Q" not in row or "A(detail)" not in row:
                        logger.error("csv file not match the format.")
                        gr.Warning("csv檔案格式不符合，略過此檔案。")
                        break
                        
                    if row["Q"] in answer_csv["Q"].values:
                        answer_csv.loc[answer_csv["Q"] == row["Q"], ["A(detail)"]] = [row["A(detail)"]]
                        if "A(summary)" in row:
                            answer_csv.loc[answer_csv["Q"] == row["Q"], ["A(summary)"]] = [row["A(summary)"]]
                        answer_csv.to_csv("./standard_response.csv", index=False)
                        continue
                    else:
                        data = {
                            "Q": row["Q"],
                            "A(detail)": row["A(detail)"],
                        }
                        if "A(summary)" in row:
                            data["A(summary)"] = row["A(summary)"]
                    
                        answer_csv = pd.concat([answer_csv, pd.DataFrame(data, index=[0])], ignore_index=True)
                        answer_csv.to_csv("./standard_response.csv", index=False)
                yield "上傳成功"
                config_info["uploaded_file"].append(full_file_name)
                json.dump(config_info, open("config.json", "w", encoding="utf-8"))
                continue
            
            else:
                logger.error(f"File {full_file_name} is not support. Ignore it...")
                gr.Warning(f"檔案 {full_file_name} 為不支援格式，略過此檔案。")
                yield f"檔案 {full_file_name} 為不支援格式，略過此檔案。"
                continue

            # 讀取pdf檔案
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for index in range(len(reader.pages)):
                    document_prefix = f"""**********\n[段落資訊]\n檔案名稱:"{file_name}{file_extension}"\n頁碼:{index+1}\n**********\n\n"""
                    text = reader.pages[index].extract_text().replace(
                        "Compal Confidentail", "")
                    encrypt_string = json.loads(
                        open("./config.json", "r", encoding="utf-8").read())["encrypt_string"]
                    if re.search(encrypt_string, text):
                        gr.Warning(f"找到異常文字 {index+1}， 略過此頁...")
                        continue

                    for index in range(len(embedding_model_list)):
                        qdrant_client.set_model(
                            embedding_model_list[index], cache_dir="./.cache")
                        qdrant_client.add(
                            collection_name=embedding_model_list[index].replace(
                                "/", "_"),
                            documents=[document_prefix + text],
                            metadata=[{"filename": full_file_name}]
                        )

            logger.info(f"Upload {full_file_name} success.")
            config_info["uploaded_file"].append(full_file_name)
            json.dump(config_info, open("config.json", "w", encoding="utf-8"))

        gr.Info("上傳成功")
        yield "上傳成功"

    def save_answer(choice: str, text_dropdown: str, detail_output_box: list, summary_output_box: list, 
                    embed_model: str, topk: str, score_threshold: float, temperature: str):
        if choice == "Yes":
            if len(detail_output_box) == 0:
                return False
            csv = pd.read_csv("./standard_response.csv", on_bad_lines='skip')
            if detail_output_box[-1][0] in csv["Q"].values:
                yield gr.update(visible=False), ""
                return True
            data = {
                "Q": detail_output_box[-1][0],
                "A(detail)": detail_output_box[-1][1],
                "A(summary)": summary_output_box[-1][1]}
            if detail_output_box[-1][0] in csv["Q"].values:
                csv.loc[csv["Q"] == detail_output_box[-1][0], ["A(detail)", "A(summary)"]] = [
                    detail_output_box[-1][1], summary_output_box[-1][1]]
            else:
                csv = pd.concat(
                    [csv, pd.DataFrame(data, index=[0])], ignore_index=True)
            csv.to_csv("./standard_response.csv", index=False)
            logger.info("Save answer success.")
            yield gr.update(visible=False), "已儲存回答."
        else:
            if detail_output_box:
                return True
            yield gr.update(visible=False), "", detail_output_box, summary_output_box, "產生新的答案..."
            for _, detail, summary, _, _ in LLM.send_query(
                text_dropdown,
                detail_output_box[-1][0],
                detail_output_box,
                summary_output_box,
                embed_model,
                topk,
                score_threshold,
                temperature,
                match_bool=False
            ):
                yield gr.update(), "", detail, summary, gr.update()
            yield gr.update(visible=True), "", detail, summary, gr.update()

    def filelist_show():
        config_info = json.load(open("config.json", "r", encoding="utf-8"))
        return config_info["uploaded_file"]

    def filelist_refresh():
        config_info = json.load(open("config.json", "r", encoding="utf-8"))
        yield gr.update(choices=config_info["uploaded_file"])
        return True

    def qdrant_delete_points(file_list: list, comfirm_checkbox: bool):
        if not comfirm_checkbox:
            gr.Warning("請勾選確認移除")
            return False

        config_info = json.load(open("config.json", "r", encoding="utf-8"))
        for file_name in file_list:
            yield gr.update(value=False), gr.update()
            for embed_model in embedding_model_list:
                qdrant_client.delete(
                    collection_name=embed_model.replace("/", "_"),
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="filename",
                                    match=models.MatchValue(value=file_name),
                                ),
                            ],
                        )
                    )
                )
            config_info["uploaded_file"].remove(file_name)
            json.dump(config_info, open("config.json", "w", encoding="utf-8"))

        yield gr.update(), gr.update(choices=File_process.filelist_show(), value=[])
        gr.Info("刪除成功")
        return True

    def dataframe_show():
        if not os.path.exists("./standard_response.csv"):
                pd.DataFrame(columns=["Q", "A(detail)", "A(summary)"]).to_csv(
                    "./standard_response.csv", index=False)
        return pd.read_csv("./standard_response.csv")

    def dataframe_refresh():
        yield gr.update(value=File_process.dataframe_show())
        return True

    def dataframe_on_select(gr_dataframe: gr.DataFrame, evt: gr.SelectData):
        yield evt.value, evt.index[0], evt.index[1], gr.update(visible=True), gr.update(visible=True, value=f"已選擇第{str(int(evt.index[0])+1)}列第{str(int(evt.index[1])+1)}欄")
        return True

    def dataframe_save_csv(gr_dataframe: gr.DataFrame, edited_textbox: str,
                           select_row: str, select_col: str):
        gr_dataframe.iat[int(select_row), int(select_col)] = edited_textbox
        gr_dataframe.to_csv("./standard_response.csv", index=False)
        yield "", gr.update(value=File_process.dataframe_show()), gr.update(value=False)
        return True

    def dataframe_add_row(gr_dataframe: gr.DataFrame):
        new_row_data = {
            "Q": "",
            "A(detail)": "",
            "A(summary)": ""
        }
        gr_dataframe = pd.concat(
            [gr_dataframe, pd.DataFrame([new_row_data])], ignore_index=True)
        gr_dataframe.to_csv("./standard_response.csv", index=False)
        yield gr.update(value=File_process.dataframe_show())
        return True
    
    def dataframe_del_row(gr_dataframe: gr.DataFrame, select_row: str, select_col: str):
        if select_row == "":
            return False
        gr_dataframe = gr_dataframe.drop(int(select_row))
        gr_dataframe.to_csv("./standard_response.csv", index=False)
        yield gr.update(value=File_process.dataframe_show()), gr.update(visible=False), gr.update(visible=False), "", ""
        return True

