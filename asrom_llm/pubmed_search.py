import os
import urllib

import pandas as pd
import psycopg2
from Bio import Entrez, Medline
from pubmed_parser import parse_medline_xml

from asrom_llm.utils import verbose_print


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


def connect_to_postgres(dbname, user, password, host, port):
    conn = psycopg2.connect(
        dbname=dbname, user=user, password=password, host=host, port=port
    )
    return conn


def insert_into_postgres(
    conn, data_dict, table_name="asrom_db", verbose=False
):
    cursor = conn.cursor()

    # Create the INSERT query
    columns = ", ".join(data_dict.keys())
    values = ", ".join(["%s"] * len(data_dict))
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"

    # Execute the query
    cursor.execute(query, list(data_dict.values()))

    # Commit the transaction
    conn.commit()

    # Close the cursor
    cursor.close()

    verbose_print("Data inserted successfully!", verbose=verbose)


gz_file = download_pubmed_baseline(
    year=23, file_num=1, dir_path="./data/pubmed/"
)
parsed_data = parse_medline_xml(gz_file)
conn = connect_to_postgres(
    dbname="postgres",
    user="postgres",
    password="password",
    host="localhost",
    port="5432",
)
