"""Download + CPU-verify the math128 R1-Distill-7B dump (32,768 samples)."""
import sys, yaml, re
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/Users/bulutyigit/Documents/pycharm_projects/how_models_reason")
from huggingface_hub import snapshot_download
from reasonbench.verification import verify_answer

REPO = "nishadsinghi/math128_solutions_r1_distill_qwen_7b_32K_tokens"
local = snapshot_download(REPO, repo_type="dataset")
print("downloaded to", local, flush=True)

def boxed(text):
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    if not starts: return None
    start = starts[-1]; depth = 1; index = start
    while index < len(text) and depth:
        if text[index] == "{": depth += 1
        elif text[index] == "}": depth -= 1
        index += 1
    return text[start:index-1]

rows = []
for path in sorted(Path(local).glob("*.yaml"), key=lambda p: int(p.stem)):
    d = yaml.safe_load(path.read_text())
    ref = boxed(d["gt_answer"])
    if ref is None:
        print("no boxed ref:", path.stem); continue
    for i, sample in enumerate(d["samples"]):
        v = verify_answer(sample, ref, "math")
        rows.append({"problem": int(path.stem), "sample": i,
                     "correct": bool(v.correct), "chars": len(sample)})
    if int(path.stem) % 16 == 0:
        print("problem", path.stem, "done", flush=True)
f = pd.DataFrame(rows)
out = Path("/Users/bulutyigit/Documents/pycharm_projects/how_models_reason/artifacts/external/math128_distill7b")
out.mkdir(parents=True, exist_ok=True)
f.to_parquet(out / "sample_correctness.parquet", index=False)
p = f.groupby("problem").correct.mean()
print("samples:", len(f), "| overall pass rate:", round(f.correct.mean(), 4))
print("problem pass-rate distribution: min", round(p.min(),3), "| q25", round(p.quantile(.25),3),
      "| median", round(p.median(),3), "| q75", round(p.quantile(.75),3), "| max", round(p.max(),3))
print("problems all-fail:", int((p==0).sum()), "| all-pass:", int((p==1).sum()), "| intermediate:", int(((p>0)&(p<1)).sum()))
