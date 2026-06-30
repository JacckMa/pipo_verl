#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


class RunCodeRequest(BaseModel):
    compile_timeout: int = 10
    run_timeout: int = 10
    code: str
    stdin: str | None = None
    memory_limit_MB: int = 1024
    language: str = "python"
    files: dict[str, str] = {}
    fetch_files: list[str] = []


app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run_code")
def run_code(req: RunCodeRequest) -> dict[str, Any]:
    if req.language not in {"python", "python3"}:
        return {
            "status": "Failed",
            "compile_result": {"status": "Error", "return_code": 1, "stderr": f"unsupported language: {req.language}"},
            "run_result": None,
        }

    with tempfile.TemporaryDirectory(prefix="hacpo_sandbox_") as tmp:
        code_path = os.path.join(tmp, "main.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(req.code)

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, code_path],
                input=req.stdin or "",
                text=True,
                capture_output=True,
                cwd=tmp,
                timeout=max(1, int(req.run_timeout)),
            )
            duration = time.time() - start
            run_status = "Finished"
            status = "Success" if proc.returncode == 0 else "Failed"
            return {
                "status": status,
                "compile_result": {"status": "Finished", "return_code": 0, "stderr": "", "execution_time": 0.0},
                "run_result": {
                    "status": run_status,
                    "return_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "execution_time": duration,
                },
            }
        except subprocess.TimeoutExpired as e:
            duration = time.time() - start
            return {
                "status": "Failed",
                "compile_result": {"status": "Finished", "return_code": 0, "stderr": "", "execution_time": 0.0},
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "return_code": -1,
                    "stdout": e.stdout or "",
                    "stderr": e.stderr or "timeout",
                    "execution_time": duration,
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
