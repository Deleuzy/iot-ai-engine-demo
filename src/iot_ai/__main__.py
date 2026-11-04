"""Command line: `python -m iot_ai run | eval-engine | serve`."""

from __future__ import annotations

import json
import sys

from iot_ai.config import RESULTS_DIR


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else ""
    if command == "run":
        from iot_ai.pipeline import run

        summary = run()
        print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))
        return 0
    if command == "eval-engine":
        from iot_ai.ai.evaluate import run_eval

        report = run_eval()
        print(
            json.dumps(
                {k: report[k] for k in ("retrieval", "guardrail_catch_rate", "guardrail_caught", "guardrail_cases")},
                indent=2,
                default=str,
            )
        )
        print(f"written to {RESULTS_DIR / 'engine_eval.json'}")
        return 0
    if command == "serve":
        import uvicorn

        uvicorn.run("iot_ai.api:create_app", factory=True, host="0.0.0.0", port=8000)
        return 0
    print("usage: python -m iot_ai run | eval-engine | serve")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
