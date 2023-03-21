import time

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pubmed_parser import parse_medline_xml
from pymilvus import connections, utility

from asrom_llm.database import ModifiedMilvus
from asrom_llm.pubmed_search import (
    download_pubmed_baseline,
    split_parsed_data_into_doc_format,
)

YEAR = 23
START_FILE_NUM = 1
MAX_FILE_NUM = 1166
HF_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
RECURSIVE_TEXT_SPLITTER_CHUNK_SIZE = 1000

embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=RECURSIVE_TEXT_SPLITTER_CHUNK_SIZE, chunk_overlap=0
)

overall_start_time = time.time()

connections.connect(alias="default", host="localhost", port="19530")
db = None
for file_num in range(START_FILE_NUM, MAX_FILE_NUM + 1):
    print(f"  Starting file {file_num}", end=" ")
    start_time = time.time()

    gz_file = download_pubmed_baseline(
        year=YEAR, file_num=file_num, dir_path="./data/pubmed/"
    )

    parsed_data = parse_medline_xml(gz_file)
    parsed_data = split_parsed_data_into_doc_format(parsed_data)

    docs = [
        Document(page_content=abstract, metadata=page_content)
        for abstract, page_content in parsed_data
    ]
    docs = text_splitter.split_documents(docs)

    if db is None:
        if utility.has_collection("medline_collection"):
            db = ModifiedMilvus(
                embedding_function=embeddings,
                connection_args={"host": "127.0.0.1", "port": "19530"},
                collection_name="medline_collection",
                text_field="text_field",
            )
        else:
            db = ModifiedMilvus.from_documents(
                docs,
                embedding=embeddings,
                connection_args={"host": "127.0.0.1", "port": "19530"},
            )
    else:
        db.add_documents(docs)

    print(f"Took {(time.time() - start_time)/60:.2f} minutes")
print(
    f"Addition of Pubmed Medline Baseline took {(time.time() - overall_start_time) / 60 / 60: .2f} hours"
)

# from pymilvus import connections, utility
# from pymilvus import Collection
# connections.connect(alias="default", host="localhost", port="19530")
# utility.list_collections()
# collection = Collection("medline_collection")
# for col in utility.list_collections():
#     # delete collection
#     utility.drop_collection(col)
