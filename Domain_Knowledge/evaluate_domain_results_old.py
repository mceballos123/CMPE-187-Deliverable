"""Legacy evaluator reference; use evaluate_domain_results.py for current results."""
import argparse, csv, json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "results" / "raw"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

def grade(response, expected, model):
    prompt = f"""You are grading one CMPE 187 Domain Knowledge AI test.

EXPECTED:
{expected}

ACTUAL:
{response}

Return JSON only:
{{"result":"PASS|PARTIAL|FAIL","reason":"one short sentence"}}

PASS = correct and satisfies expected behavior.
PARTIAL = generally correct but misses an important required behavior/detail.
FAIL = significant technical error, irrelevant answer, fabricated information, or failure to perform required behavior.
"""
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model":model,"prompt":prompt,"stream":False,"format":"json"},
                      timeout=180)
    r.raise_for_status()
    data = json.loads(r.json()["response"])
    result = data.get("result","FAIL").upper()
    if result not in {"PASS","PARTIAL","FAIL"}:
        result = "FAIL"
    return result, data.get("reason","")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3.1:8b")
    args = p.parse_args()

    rows = []
    for app in ["chatgpt","claude","gemini","perplexity"]:
        f = RAW_DIR / f"{app}.json"
        if f.exists():
            for rec in json.loads(f.read_text()):
                result, reason = grade(rec["response"], rec["expected"], args.model)
                rows.append({**rec,"result":result,"reason":reason})
                print(app, rec["id"], result, "-", reason)

    if not rows:
        raise SystemExit("No raw files found. Run evaluation_results.py first.")

    detail = OUT_DIR / "domain_knowledge_results.csv"
    with detail.open("w", newline="", encoding="utf-8") as f:
        fields = ["app","id","prompt","expected","result","reason","response"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = OUT_DIR / "domain_knowledge_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["AI Application","Executed","Passed","Partial Pass","Failed","Full Pass Rate"])
        for app in ["chatgpt","claude","gemini","perplexity"]:
            data = [r for r in rows if r["app"] == app]
            pcount = sum(r["result"]=="PASS" for r in data)
            partial = sum(r["result"]=="PARTIAL" for r in data)
            fail = sum(r["result"]=="FAIL" for r in data)
            rate = (pcount/len(data)*100) if data else 0
            w.writerow([app,len(data),pcount,partial,fail,f"{rate:.3f}%"])

    print("\nCreated:")
    print(detail)
    print(summary)

if __name__ == "__main__":
    main()
