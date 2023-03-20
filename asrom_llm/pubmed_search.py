import os
import urllib

import pandas as pd
from Bio import Entrez, Medline
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pubmed_parser import parse_medline_xml


def search_pubmed(query, page_num=1, page_size=10):
    """
    Searches the Pubmed database for articles that match the given search query,
    and returns the results in the specified page.

    Args:
    - search_term (str): The term to search for in the Pubmed database.
    - page_num (int): The page number to retrieve.
    - page_size (int): The number of results per page.

    Returns:
    - generator: A generator of dictionaries, each representing a Medline record.
    """
    Entrez.email = "info@asrom.org"  # Replace with your email address

    # Use the esearch function to search the Pubmed database with the given search term.
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retstart=page_size * (page_num - 1),
        retmax=page_size,
    )
    record = Entrez.read(handle)

    # Extract the list of article IDs from the search results.
    id_list = record["IdList"]

    # Use the efetch function to retrieve the full Medline records for the articles.
    handle = Entrez.efetch(
        db="pubmed", id=id_list, rettype="medline", retmode="text"
    )
    records = Medline.parse(handle)
    return records


def parse_result(records):
    """
    Parses the relevant data points (abstracts, authors, and publication dates)
    from a list of Medline records.

    Args:
    - records (list): A list of dictionaries, each representing a Medline record.

    Returns:
    - pandas.DataFrame: A DataFrame with columns "Abstract", "Authors", and "Publication Date".
    """
    # Use a list comprehension to extract the relevant data points from each record.
    results = [
        {
            "Abstract": record.get("AB", ""),
            "Authors": record.get("AU", ""),
            "Publication Date": record.get("DP", ""),
            "Title": record.get("TI", ""),
            "Journal Title": record.get("JT", ""),
            "PMC": record.get("PMC", ""),
            "PMID": record.get("PMID", ""),
        }
        for record in records
    ]

    # Convert the list of dictionaries to a DataFrame.
    df = pd.DataFrame(results)
    return df


def download_pubmed_baseline(year=23, file_num=1, dir_path="./data/pubmed/"):
    """
    Downloads the Pubmed baseline for the given year.

    Args:
    - year (int): The year to download the baseline for.
    - file_num (int): The number of the file to download.
    - dir_path (str): The directory path to download the file to.

    Returns:
    - str: The path to the downloaded file.
    """
    filename = f"pubmed{year}n{str(file_num).rjust(4, '0')}.xml.gz"
    url = f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{filename}"
    file_path = os.path.join(dir_path, filename)

    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    return file_path


def split_parsed_data_into_doc_format(parsed_data):
    """Splits the parsed data into a list of tuples in doc format

    Takes the parsed data (list of dictionaries) and splits it into the
    doc format of an array of tuples where the first element is the abstract
    and the second is the metadata, without the abstract, and with a source

    Args:
    - parsed_data (list): A list of dictionaries, each representing a Medline record.

    Returns:
    - list: A list of tuples in doc format.
    """
    result = []
    for data_dict in parsed_data:
        if data_dict["delete"] is True or not data_dict["abstract"]:
            continue
        else:
            abstract = data_dict["abstract"]
            del data_dict["abstract"]
            data_dict["source"] = data_dict["pmid"]
            data_dict = {k: str(v) for k, v in data_dict.items()}
            result.append((abstract, data_dict))
    return result


gz_file = download_pubmed_baseline(
    year=23, file_num=1, dir_path="./data/pubmed/"
)
parsed_data = parse_medline_xml(gz_file)
parsed_data = split_parsed_data_into_doc_format(parsed_data)
docs = [
    Document(page_content=abstract, metadata=page_content)
    for abstract, page_content in parsed_data
]
HF_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
vector_db = Chroma(
    embedding_function=embeddings,
    client_settings=Settings(
        chroma_api_impl="rest",
        chroma_server_host="localhost",
        chroma_server_http_port="8000",
    ),
)

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    client_settings=Settings(
        chroma_api_impl="rest",
        chroma_server_host="localhost",
        chroma_server_http_port="8000",
    ),
)
