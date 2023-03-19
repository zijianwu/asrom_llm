import pandas as pd
from Bio import Entrez, Medline


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
