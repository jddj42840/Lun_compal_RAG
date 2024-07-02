from utils.llm import LLM
from utils.file_process import File_process
from utils.qdrant import Qdrant, embedding_model_list
import gradio as gr

css = """
.btn-refresh {
    max-width: 2.7em;
    min-width: 2.7em;
    height: 39.594px;
    align-self: end;
    line-height: 1em;
    border-radius: 0.5em;
    flex: none;
    padding: 0px;
}

.button-icon {
    margin:0px;
}

.edit-csv-new-row-btn {
    height: 40px;
    max-width: 6em;
    min-width: 6em;
    padding: 8px;
    align-self: start;
}

"""


def advanced_checkbox_change(advanced_checkbox: bool):
    yield gr.update(visible=True) if advanced_checkbox else gr.update(visible=False)


def function_dropdown_change(function_dropdown: str):
    if function_dropdown == "修改已儲存的正確答案":
        yield gr.update(visible=True), gr.update(visible=False)
    elif function_dropdown == "移除已上傳的文件":
        yield gr.update(visible=False), gr.update(visible=True)
    else:
        yield gr.update(visible=False), gr.update(visible=False)


def edit_csv_checkbox_change(checkbox_status: bool):
    if checkbox_status:
        yield gr.update(visible=True)
    else:
        yield gr.update(visible=False)


def rm_file_checkbox_change(checkbox_status: bool):
    if checkbox_status:
        yield gr.update(visible=True)
    else:
        yield gr.update(visible=False)


with gr.Blocks() as chat:
    with gr.Row():
        with gr.Column():
            with gr.Column(visible=False) as resp_row:
                gr.Markdown("### Is the answer correct?")
                with gr.Row():
                    yes_btn = gr.Button(value="Yes")
                    no_btn = gr.Button(value="No")

            upload = gr.Files(label="Upload file")
            msg_box = gr.Textbox(label="輸入問題")
            advanced_checkbox = gr.Checkbox(
                label="Advanced Options", value=False)
            submit_btn = gr.Button(value="送出")
            clear_button = gr.ClearButton(value="清除")

            with gr.Column(visible=False) as advanced_block:
                gr.Markdown("### RAG 參數設置")
                embed_model_dropdown = gr.Dropdown(
                    label="Embedding Model", 
                    choices=embedding_model_list, 
                    value="intfloat/multilingual-e5-large",
                    interactive=True, 
                    info="詞嵌入模型，主要影響相似度搜尋準確性。"
                )
                topk = gr.Textbox(label="Top-K", 
                    value="5",
                    info="搜尋指定筆數資料給語言模型做資料彙整，建議值為5~8"
                )
                score_threshold = gr.Textbox(
                    label="Score Threshold", 
                    value="0", 
                    visible=True,
                    info="相似度門檻閥值，超過閥值分數之資料才會被列入搜尋結果。multilingual-e5-large建議閥值為0.84~0.86, bge-small-zh-v1.5建議閥值為0.51。 0為不設定閥值"
                )
                gr.Markdown("### 語言模型參數設置")
                text_model_dropdown = gr.Dropdown(
                    label="Language Model", 
                    choices=LLM.get_model_list(), 
                    value=lambda: LLM.get_model(),
                    interactive=True, 
                    info="語言模型，負責將相似度搜尋結果資訊做彙整並且輸出成詳細+摘要段落"
                )
                temperature = gr.Textbox(
                    label="Temperature", 
                    value="0", 
                    visible=True,
                    info="溫度參數，控制生成文本的多樣性。 0為固定輸出"
                )
                prompt = gr.Textbox(label="prompt",  max_lines=100, visible=True)
            status = gr.Markdown(value="")

        detail_output_box = gr.Chatbot(label="Detail output:", height=800)
        summary_output_box = gr.Chatbot(label="Summary output:", height=800)

    submit_btn.click(
        LLM.send_query, 
        inputs=[
            text_model_dropdown, 
            msg_box, 
            detail_output_box, 
            summary_output_box, 
            embed_model_dropdown,
            topk, 
            score_threshold, 
            temperature, 
            prompt
        ], 
        outputs=[msg_box, detail_output_box, summary_output_box, resp_row, status]
    )
    upload.upload(File_process.load_file, inputs=[upload], outputs=[status])
    clear_button.add([detail_output_box, summary_output_box, msg_box])
    yes_btn.click(
        File_process.save_answer,
        inputs=[
            yes_btn, 
            text_model_dropdown, 
            detail_output_box,
            summary_output_box, 
            embed_model_dropdown, 
            topk, 
            score_threshold, 
            temperature
        ],
        outputs=[resp_row, status]
    )
    no_btn.click(
        File_process.save_answer,
        inputs=[
            no_btn, 
            text_model_dropdown, 
            detail_output_box, 
            summary_output_box, 
            embed_model_dropdown,
            topk, 
            score_threshold, 
            temperature
        ],
        outputs=[resp_row, msg_box, detail_output_box, summary_output_box, status]
    )
    advanced_checkbox.change(
        advanced_checkbox_change, 
        inputs=[advanced_checkbox], 
        outputs=[advanced_block]
    )

with gr.Blocks() as management:
    with gr.Row():
        with gr.Column():
            function_dropdown = gr.Dropdown(label="請選擇一項功能", allow_custom_value=False,
                                            choices=["修改已儲存的正確答案", "移除已上傳的文件"], interactive=True)

    # 管理歷史上傳文件
    with gr.Column(visible=False) as collection_manage:
        with gr.Row():
            with gr.Column():
                rm_file_dropdown = gr.Dropdown(label="已上傳文件", choices=File_process.filelist_show(),
                                               multiselect=True)
            with gr.Column():
                with gr.Row():
                    rm_file_refresh_btn = gr.Button(
                        value="", 
                        icon="./icons/refresh.png", 
                        interactive=True, 
                        elem_classes="btn-refresh"
                    )
                    rm_file_checkbox = gr.Checkbox(label="確認移除", value=False)
                with gr.Row():
                    rm_file_delete_btn = gr.Button(value="移除", interactive=True, visible=False)

    # 管理已儲存的正確答案
    with gr.Column(visible=False) as edit_standard_response:
        with gr.Row():
            edit_csv_dataframe = gr.DataFrame(
                value=lambda: File_process.dataframe_show(), 
                height=400, 
                column_widths=["30%"],
                wrap=True, 
                interactive=False
            )
        with gr.Row():
            edit_csv_add_row_btn = gr.Button(value="新增一列", elem_classes="edit-csv-new-row-btn")
            edit_csv_del_row_btn = gr.Button(value="刪除選擇", elem_classes="edit-csv-new-row-btn")
        with gr.Row():
            edit_csv_status = gr.Markdown(value="")
        with gr.Row(visible=False) as edit_section:
            with gr.Row():
                edit_csv_textbox = gr.TextArea(label="編輯框", value="", max_lines=10)
            with gr.Row(visible=False):
                edit_csv_row = gr.Textbox(value="")
                edit_csv_col = gr.Textbox(value="")
            with gr.Row():
                with gr.Column():
                    edit_csv_checkbox = gr.Checkbox(label="確認修改", value=False)
                    edit_csv_save_btn = gr.Button(value="儲存", visible=False)

    function_dropdown.change(
        function_dropdown_change, 
        inputs=[
        function_dropdown], 
        outputs=[edit_standard_response, collection_manage]
    )
    # rm_file
    rm_file_refresh_btn.click(File_process.filelist_refresh, outputs=[rm_file_dropdown])
    rm_file_delete_btn.click(
        File_process.qdrant_delete_points, 
        inputs=[rm_file_dropdown, rm_file_checkbox],
        outputs=[rm_file_checkbox, rm_file_dropdown])
    rm_file_checkbox.change(
        rm_file_checkbox_change, 
        inputs=[rm_file_checkbox], 
        outputs=[rm_file_delete_btn]
    )
    # edit csv
    edit_csv_dataframe.select(
        File_process.dataframe_on_select, 
        inputs=[edit_csv_dataframe], 
        outputs=[
            edit_csv_textbox, 
            edit_csv_row, 
            edit_csv_col, 
            edit_section, 
            edit_csv_status
            ]
        )
    edit_csv_checkbox.change(
        edit_csv_checkbox_change, 
        inputs=[edit_csv_checkbox], 
        outputs=[edit_csv_save_btn])
    edit_csv_add_row_btn.click(
        File_process.dataframe_add_row, 
        inputs=[edit_csv_dataframe],
        outputs=[edit_csv_dataframe])
    edit_csv_save_btn.click(
        File_process.dataframe_save_csv,
        inputs=[edit_csv_dataframe, edit_csv_textbox, edit_csv_row, edit_csv_col],
        outputs=[edit_csv_textbox, edit_csv_dataframe, edit_csv_checkbox])
    edit_csv_del_row_btn.click(
        File_process.dataframe_del_row, 
        inputs=[edit_csv_dataframe, edit_csv_row, edit_csv_col], 
        outputs=[edit_csv_dataframe, edit_section, edit_csv_status, edit_csv_row, edit_csv_col]
    )


if __name__ == "__main__":
    Qdrant.qdrant_start_db()
    app = gr.TabbedInterface(
        interface_list=[chat, management], 
        tab_names=["聊天", "管理"],
        title="Compal RAG", 
        css=css
    )
    app.queue(max_size=None)
    app.launch(server_port=7861, server_name="0.0.0.0")
