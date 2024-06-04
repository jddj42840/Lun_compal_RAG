import os
import PyPDF2
import pypdfium2 as pdfium
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

"""初始化qdrant連接"""
qdrant_client = QdrantClient(url="http://10.147.17.3:6333")
pdf_file = "pdf_output/A32_QCU00_CNC-D-Cut-on-3D-Metal-Forming-Surface_Lesson-Learnt_2012.pdf"
file_extension = os.path.splitext(pdf_file)[1]

"""設定fastembed模型"""
# qdrant_client.set_model("intfloat/multilingual-e5-large", cache_dir="./.cache")

"""設定SentenceTransformer模型"""
# embedding_model = SentenceTransformer("GanymedeNil/text2vec-large-chinese", device="cpu", cache_folder="./.cache")
embedding_model = SentenceTransformer("./modelscope/text2vec-large-chinese", device="cpu", cache_folder="./.cache")

"""預計新增AutoTokenizer, AutoModel的方法"""
# code here

"""暫時存放extract_text的資料"""
data_arr = []

"""(re)create collection"""
# qdrant_client.recreate_collection(
#     collection_name="test_rag", 
#     # vectors_config=VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
#     vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE, on_disk=True),
#     optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
#     quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
#         type=models.ScalarType.INT8)),
#     hnsw_config=models.HnswConfigDiff(on_disk=True))

"""pypdfium2 extract text"""
# file = pdfium.PdfDocument(pdf_file)
# temp = ""
# for index in range(len(file)):
#     text = file[index].get_textpage().get_text_range(errors="strict")
#     document_prefix = f"""**********\n[段落資訊]\n檔案名稱:"{file_name}{file_extension}"\n頁碼:{index+1}\n**********\n\n"""
#     temp += document_prefix + text + "\n\n"

# open("text.txt", "w").write(temp)

"""pypdf2 extract text"""
for pdf_file in os.listdir("pdf_output/test_pdf"):
    with open("pdf_output/test_pdf/" + pdf_file, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        temp = ""
        for index in range(len(reader.pages)):
            document_prefix = f"""**********\n[段落資訊]\n檔案名稱:"{file_name}{file_extension}"\n頁碼:{index+1}\n**********\n\n"""
            doc = reader.pages[index].extract_text()
            data_arr.append(document_prefix + doc + "\n\n")


"""add document with fastembed method"""
# qdrant_client.add(
#     collection_name="test_rag",
#     documents=[temp],
#     metadata=[{"keywords": ['如何', '減少', '鋁蓋', '側邊', '因', '正反', '正反面', '反面', '噴砂', '各', '一次', '的', '壓力', '過', '大', '而', '造成', '的', '凹坑', '？']}])

"""add document with SentenceTransformer method"""
# for index in range(len(data_arr)):
#     qdrant_client.upload_collection(
#         collection_name="test_rag",
#         vectors=[embedding_model.encode(data_arr[index])],
#         payload=[{"document": data_arr[index]}]
#     )


"""非使用fastembed搜尋資料的方法"""
result = qdrant_client.search(
    collection_name="test_rag",
    # query_vector=embedding_model.encode("藍芽"),
    query_vector=embedding_model.encode("如何減少鋁蓋側邊因正反面噴砂各一次的壓力過大而造成的凹坑？"),
    query_filter=None,
    with_payload=True,
    limit=5)

with open("result.txt", "w") as f:
    f.write(str(result))

"""關鍵字搜尋"""
# response = qdrant_client.scroll( 
#     collection_name="test_rag",
#     scroll_filter=models.Filter(
#         should=[
#             models.FieldCondition(
#                 key="keywords",
#                 match=models.MatchAny(
#                     any=jieba.lcut(text, cut_all=True)
#                 )
#             )
#         ]
#     )
# )

# for hit in response:
#     print(hit)
# print(len(response))

qdrant_client.close()