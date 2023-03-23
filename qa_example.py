import os
from collections import defaultdict

from asrom_llm.qa import (
    get_qa_v1,
    get_qa_v2,
    get_qa_v3,
    get_qa_v4,
    get_qa_v5,
    get_qa_v6,
    get_qa_v7,
    get_qa_v8,
    get_qa_v9,
    get_qa_v10,
    get_qa_v11,
    get_qa_v12,
    process_query,
)
from asrom_llm.utils import get_key_from_value, load_json, save_json

QUERIES = {
    "What are some non‐pharmacological interventions for headache in children and adolescents?": None,
    "What are the indications for spontaneous bacterial peritonitis prophylaxis?": None,
    "What are the reasons to start antibiotics to prevent spontaneous bacterial peritonitis prophylaxis?": [
        get_qa_v1
    ],  # Bad question
    "Does psilocybin cause psychosis?": None,
    "How do you diagnose and treat superior mesenteric artery syndrome?": None,  # Does not give results in Pubmed due to odd Pubmed search issue
}
QA_FUNCTIONS = {
    "v1": get_qa_v1,
    "v2": get_qa_v2,
    "v3": get_qa_v3,
    "v4": get_qa_v4,
    "v5": get_qa_v5,
    "v6": get_qa_v6,
    "v7": get_qa_v7,
    "v8": get_qa_v8,
    "v9": get_qa_v9,
    "v10": get_qa_v10,
    "v11": get_qa_v11,
    "v12": get_qa_v12,
}
RESULTS_PATH = "results.json"
HF_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def load_results_json(RESULTS_PATH):
    if os.path.isfile(RESULTS_PATH):
        results = load_json(RESULTS_PATH)
        results = defaultdict(lambda: defaultdict(str), results)
    else:
        results = defaultdict(lambda: defaultdict(str))
    return results


for query, functions in QUERIES.items():
    if functions is None:
        functions = QA_FUNCTIONS.values()
    for function in functions:
        results = load_results_json(RESULTS_PATH)
        version = get_key_from_value(function, QA_FUNCTIONS)
        results = process_query(
            results, query, version, function, verbose=True
        )
        save_json(results, RESULTS_PATH)
