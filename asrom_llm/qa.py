import time

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from asrom_llm.pubmed_search import parse_result, search_pubmed
from asrom_llm.utils import verbose_print


def get_qa_v1(query, llm=None, verbose=False):
    start_time = time.time()
    verbose_print("Loading LLM...", verbose=verbose, end="")
    if not llm:
        llm = OpenAI(
            model_name="gpt-3.5-turbo",
            organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
            temperature=0.3,
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


def get_qa_v2(query, llm=None, verbose=False):
    start_time = time.time()
    verbose_print("Loading LLM...", verbose=verbose, end="")
    if not llm:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
            temperature=0.5,
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


def get_qa_v3(query, llm=None, verbose=False):
    start_time = time.time()
    verbose_print("Loading LLM...", verbose=verbose, end="")
    if not llm:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
            temperature=0.5,
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
