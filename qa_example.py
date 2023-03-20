import os
from collections import defaultdict

from asrom_llm.qa import get_qa_v1, get_qa_v2, get_qa_v3
from asrom_llm.utils import load_json, save_json

QUERIES = [
    "What are some non‐pharmacological interventions for headache in children and adolescents?",
    "What are the indications for spontaneous bacterial peritonitis prophylaxis?",
    "What are the reasons to start antibiotics to prevent spontaneous bacterial peritonitis prophylaxis?",
    "Does psilocybin cause psychosis?",
]
RESULTS_PATH = "results.json"

if os.path.isfile(RESULTS_PATH):
    results = load_json(RESULTS_PATH)
    results = defaultdict(lambda: defaultdict(str), results)
else:
    results = defaultdict(lambda: defaultdict(str))

for QUERY in QUERIES:
    if not results[QUERY].get("v1", None):
        v1_result = get_qa_v1(QUERY, verbose=True)
        results[QUERY]["v1"] = v1_result
    if not results[QUERY].get("v2", None):
        v2_result = get_qa_v2(QUERY, verbose=True)
        results[QUERY]["v2"] = v2_result
    if not results[QUERY].get("v3", None):
        v3_result = get_qa_v3(QUERY, verbose=True)
        results[QUERY]["v3"] = v3_result

save_json(results, RESULTS_PATH)
