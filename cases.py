import json
import os

_cases_file = os.path.join(os.path.dirname(__file__), "cases.json")

def load_cases():
    if not os.path.exists(_cases_file):
        return []
    with open(_cases_file, "r", encoding="utf-8") as f:
        return json.load(f)

CASES = load_cases()
