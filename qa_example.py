import os
from collections import defaultdict

from asrom_llm.qa import (
    get_qa_v1,
    get_qa_v2,
    get_qa_v3,
    get_qa_v4,
    get_qa_v5,
    get_qa_v6,
    process_query,
)
from asrom_llm.utils import load_json, save_json

QUERIES = [
    "What are some non‐pharmacological interventions for headache in children and adolescents?",
    "What are the indications for spontaneous bacterial peritonitis prophylaxis?",
    "What are the reasons to start antibiotics to prevent spontaneous bacterial peritonitis prophylaxis?",
    "Does psilocybin cause psychosis?",
]
RESULTS_PATH = "results.json"
HF_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

if os.path.isfile(RESULTS_PATH):
    results = load_json(RESULTS_PATH)
    results = defaultdict(lambda: defaultdict(str), results)
else:
    results = defaultdict(lambda: defaultdict(str))

qa_functions = {
    "v1": get_qa_v1,
    "v2": get_qa_v2,
    "v3": get_qa_v3,
    "v4": get_qa_v4,
    "v5": get_qa_v5,
    "v6": get_qa_v6,
}

for QUERY in QUERIES:
    for version, function in qa_functions.items():
        process_query(results, QUERY, version, function)

save_json(results, RESULTS_PATH)
