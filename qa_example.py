import os
from collections import defaultdict

from asrom_llm.qa import get_qa_v1
from asrom_llm.utils import load_json, save_json

QUERY = "What are some non‐pharmacological interventions for headache in children and adolescents?"
RESULTS_PATH = "results.json"

if os.path.isfile(RESULTS_PATH):
    results = load_json(RESULTS_PATH)
    results = defaultdict(lambda: defaultdict(str), results)
else:
    results = defaultdict(lambda: defaultdict(str))

if not results[QUERY]["v1"]:
    v1_result = get_qa_v1(QUERY, verbose=True)
    results[QUERY]["v1"] = v1_result

save_json(results, RESULTS_PATH)
