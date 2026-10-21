"""Compare same-container benchmark files and fail on repeated slowdowns."""

import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("--limit", type=float, default=0.05)
    args = parser.parse_args()
    records = [
        json.loads(path.read_text())
        for path in (args.candidate, args.baseline, args.repeat)
    ]
    for key in ("mode", "torch", "vllm", "gpu", "torch_num_threads", "omp_num_threads"):
        assert records[0][key] == records[1][key] == records[2][key], key
    assert records[0]["worker_sha256"] == records[2]["worker_sha256"]
    assert records[0]["worker_sha256"] != records[1]["worker_sha256"], (
        "baseline loaded candidate code"
    )
    cases = [{item["case"]: item for item in result["workloads"]} for result in records]
    assert cases[0].keys() == cases[1].keys() == cases[2].keys(), "workload sets differ"
    failures = []
    print("| Case | Baseline (s) | Candidate change | Repeat change |")
    print("| --- | ---: | ---: | ---: |")
    for name, baseline in cases[1].items():
        changes = [
            cases[index][name]["median_s"] / baseline["median_s"] - 1
            for index in (0, 2)
        ]
        print(
            f"| {name} | {baseline['median_s']:.4f} | {changes[0]:+.1%} | {changes[1]:+.1%} |"
        )
        if changes[1] > args.limit:
            failures.append(name)
    print(f"Repeat slowdowns above {args.limit:.0%}: {len(failures)}")
    raise SystemExit(bool(failures))
