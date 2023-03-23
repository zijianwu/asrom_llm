import time

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from asrom_llm.pubmed_search import parse_result, search_pubmed
from asrom_llm.utils import verbose_print


def process_query(results, query, version, function):
    if not results[query].get(version, None):
        try:
            result = function(query, verbose=True)
            results[query][version] = result
        except Exception as e:
            print(f"Error processing {query} with {version}: {e}")
    return results


# Pubmed Search, Stuff, 0.2 temperature
def get_qa_v1(query, llm=None, verbose=False):
    start_time = time.time()
    verbose_print("Loading LLM...", verbose=verbose, end="")
    if not llm:
        llm = OpenAI(
            model_name="gpt-3.5-turbo",
            organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
            temperature=0.2,
        )
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Searching Pubmed...", verbose=verbose, end="")
    records = search_pubmed(query, page_num=1, page_size=8)
    parsed_result = parse_result(records)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Parsing documents...", verbose=verbose, end="")
    docs = [
        Document(
            page_content=record["Abstract"],
            metadata=record.drop("Abstract").to_dict(),
        )
        for _, record in parsed_result.iterrows()
    ]
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(docs)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Getting answer... ", verbose=verbose, end="")
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=docs, question=query)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)
    return result


# Pubmed Search, Map Reduce, 0.2 temperature
def get_qa_v2(query, llm=None, verbose=False):
    start_time = time.time()
    verbose_print("Loading LLM...", verbose=verbose, end="")
    if not llm:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
            temperature=0.2,
        )
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Searching Pubmed...", verbose=verbose, end="")
    records = search_pubmed(query, page_num=1, page_size=8)
    parsed_result = parse_result(records)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Parsing documents...", verbose=verbose, end="")
    docs = [
        Document(
            page_content=record["Abstract"],
            metadata={
                "source": record["PMID"],
                **record.drop("Abstract").to_dict(),
            },
        )
        for _, record in parsed_result.iterrows()
    ]
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(docs)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Getting answer... ", verbose=verbose, end="")
    chain = load_qa_with_sources_chain(
        llm, chain_type="map_reduce", verbose=verbose
    )
    result = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)
    return result


# Pubmed Search, Refine, 0.2 temperature
def get_qa_v3(query, llm=None, verbose=False):
    start_time = time.time()
    verbose_print("Loading LLM...", verbose=verbose, end="")
    if not llm:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
            temperature=0.2,
        )
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Searching Pubmed...", verbose=verbose, end="")
    records = search_pubmed(query, page_num=1, page_size=8)
    parsed_result = parse_result(records)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Parsing documents...", verbose=verbose, end="")
    docs = [
        Document(
            page_content=record["Abstract"],
            metadata={
                "source": record["PMID"],
                **record.drop("Abstract").to_dict(),
            },
        )
        for _, record in parsed_result.iterrows()
    ]
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(docs)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Getting answer... ", verbose=verbose, end="")
    chain = load_qa_with_sources_chain(
        llm, chain_type="refine", verbose=verbose
    )
    result = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)
    return result


# Pubmed Search, Stuff, 0.8 temperature
def get_qa_v4(query, llm=None, verbose=False):
    start_time = time.time()
    verbose_print("Loading LLM...", verbose=verbose, end="")
    if not llm:
        llm = OpenAI(
            model_name="gpt-3.5-turbo",
            organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
            temperature=0.8,
        )
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Searching Pubmed...", verbose=verbose, end="")
    records = search_pubmed(query, page_num=1, page_size=8)
    parsed_result = parse_result(records)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Parsing documents...", verbose=verbose, end="")
    docs = [
        Document(
            page_content=record["Abstract"],
            metadata=record.drop("Abstract").to_dict(),
        )
        for _, record in parsed_result.iterrows()
    ]
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(docs)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Getting answer... ", verbose=verbose, end="")
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=docs, question=query)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)
    return result


# Pubmed Search, Map Reduce, 0.8 temperature
def get_qa_v5(query, llm=None, verbose=False):
    start_time = time.time()
    verbose_print("Loading LLM...", verbose=verbose, end="")
    if not llm:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
            temperature=0.8,
        )
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Searching Pubmed...", verbose=verbose, end="")
    records = search_pubmed(query, page_num=1, page_size=8)
    parsed_result = parse_result(records)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Parsing documents...", verbose=verbose, end="")
    docs = [
        Document(
            page_content=record["Abstract"],
            metadata={
                "source": record["PMID"],
                **record.drop("Abstract").to_dict(),
            },
        )
        for _, record in parsed_result.iterrows()
    ]
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(docs)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Getting answer... ", verbose=verbose, end="")
    chain = load_qa_with_sources_chain(
        llm, chain_type="map_reduce", verbose=verbose
    )
    result = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)
    return result


# Pubmed Search, Refine, 0.8 temperature
def get_qa_v6(query, llm=None, verbose=False):
    start_time = time.time()
    verbose_print("Loading LLM...", verbose=verbose, end="")
    if not llm:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
            temperature=0.8,
        )
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Searching Pubmed...", verbose=verbose, end="")
    records = search_pubmed(query, page_num=1, page_size=8)
    parsed_result = parse_result(records)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Parsing documents...", verbose=verbose, end="")
    docs = [
        Document(
            page_content=record["Abstract"],
            metadata={
                "source": record["PMID"],
                **record.drop("Abstract").to_dict(),
            },
        )
        for _, record in parsed_result.iterrows()
    ]
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(docs)
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

    start_time = time.time()
    verbose_print("Getting answer... ", verbose=verbose, end="")
    chain = load_qa_with_sources_chain(
        llm, chain_type="refine", verbose=verbose
    )
    result = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    end_time = time.time()
    verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)
    return result


# from langchain.chains import VectorDBQAWithSourcesChain
# from langchain import OpenAI
# from langchain.docstore.document import Document
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from asrom_llm.database import ModifiedMilvus


# query = "How has the challenge of Endoscopic retrograde cholangiopancreatography (ERCP) in patients with Roux-en-Y gastric bypass anatomy typically been solved?"
# verbose = True
# llm = None
# HF_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

# start_time = time.time()
# verbose_print("Loading LLM...", verbose=verbose, end="")
# if not llm:
#     llm = OpenAI(
#         model_name="gpt-3.5-turbo",
#         organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
#         temperature=0.2,
#     )
# end_time = time.time()
# verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

# start_time = time.time()
# verbose_print("Loading database and embeddings...", verbose=verbose, end="")
# embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
# db = ModifiedMilvus(
#                 embedding_function=embeddings,
#                 connection_args={"host": "127.0.0.1", "port": "19530"},
#                 collection_name="medline_collection",
#                 text_field="text_field",
#             )
# chain = VectorDBQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", vectorstore=db)
# chain.k = 8
# end_time = time.time()
# verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)

# start_time = time.time()
# verbose_print("Getting answer... ", verbose=verbose, end="")
# result = chain({"question": query}, return_only_outputs=True)
# end_time = time.time()
# verbose_print(f"{int(end_time - start_time)} secs", verbose=verbose)
