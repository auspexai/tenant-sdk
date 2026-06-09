"""auspexai_tenant.lite — the VENDORABLE stdlib executor kit (W-S step 4).

Copy this single file into your executor package. It has ZERO dependencies
beyond the Python standard library, so it runs under the worker sandbox's
system Python (no pydantic, no auspexai_tenant install) — the §9 #40b
"lite harness" answer for Phase-1 executors, plus the inference client for
worker-served models (§9 #43).

Two pieces:

- `LiteHarness(run_one)` — the stdlib mirror of `ExecutorHarness`: same
  CLI contract (`--input/--output/--models/--timeout`), same exit codes
  (0 tenant-ok / 1 tenant-failure / 2 harness-IO), same atomic output
  write. `run_one(unit, models_dir)` receives the work unit as a plain
  dict (not a pydantic model) and a `pathlib.Path`.

- `InferenceClient` — the broker client for worker-served local models.
  The worker passes the per-unit socket in `$AUSPEXAI_INFERENCE_SOCKET`
  and the authorized model id in `$AUSPEXAI_INFERENCE_MODEL`; the client
  defaults to both. Line-delimited JSON over a unix socket (the broker
  protocol in the worker's `inference/broker.py`):

      client = InferenceClient.from_env()
      reply = client.generate([{"role": "user", "content": "…"}],
                              options={"seed": 0, "num_predict": 256})
      reply["message"]["content"]   # the deterministic generation
      client.info()["gguf_sha256"]  # served-model provenance digest —
                                    # stamp it into your result payload

  Determinism: the broker FORCES temperature 0 and rejects sampling
  options — only `seed` / `num_predict` / `num_ctx` are accepted.
  Broker-side failures raise `InferenceError(code, detail)` with the
  broker's error code (`unauthorized_model`, `params_rejected`,
  `caps_exceeded`, `backend_error`, `bad_request`).

When authoring inside the SDK you can also `from auspexai_tenant.lite
import InferenceClient, LiteHarness` — the file is the same either way.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

SOCKET_ENV = "AUSPEXAI_INFERENCE_SOCKET"
MODEL_ENV = "AUSPEXAI_INFERENCE_MODEL"

# The work-unit fields the worker materializes (the SDK WorkUnit contract).
# The lite harness REQUIRES these but tolerates additive unknown fields
# (lenient reader — a newer worker must not break a vendored executor).
_REQUIRED_UNIT_FIELDS = (
    "schema_version",
    "unit_id",
    "tenant_id",
    "experiment_id",
    "manifest_sha256",
    "created_at",
    "payload",
)


class InferenceError(Exception):
    """A broker-level failure. `code` is the broker's error code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class InferenceClient:
    """Client for the worker's per-unit inference broker socket."""

    def __init__(self, socket_path: str, model: str | None = None, timeout: float = 600.0):
        self.socket_path = socket_path
        self.model = model if model is not None else os.environ.get(MODEL_ENV)
        self.timeout = timeout

    @classmethod
    def from_env(cls, timeout: float = 600.0) -> InferenceClient:
        """Construct from the env the worker sets on every inference-enabled
        unit. Raises RuntimeError when run outside an inference-enabled
        worker (e.g. the worker has `[inference] backend = "none"`)."""
        path = os.environ.get(SOCKET_ENV)
        if not path:
            raise RuntimeError(
                f"{SOCKET_ENV} is not set — this unit is not running on an inference-enabled worker"
            )
        return cls(path, timeout=timeout)

    def generate(
        self, messages: list, options: dict | None = None, model: str | None = None
    ) -> dict:
        """One deterministic chat generation against the served model.

        Returns the broker reply (`message`, `eval_count`, `model`).
        Raises InferenceError on a broker-level refusal or backend failure.
        """
        body = {
            "op": "generate",
            "model": model if model is not None else self.model,
            "messages": messages,
        }
        if options is not None:
            body["options"] = options
        return self._request(body)

    def info(self) -> dict:
        """The served model's identity + provenance (`model`, `gguf_sha256`,
        `backend_handle`). Stamp `gguf_sha256` into your result payload so
        the supply-chain digest rides into the receipt/attestation chain."""
        return self._request({"op": "info"})

    # ---- wire ---------------------------------------------------------------

    def _request(self, body: dict) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self.socket_path)
            s.sendall(json.dumps(body).encode("utf-8") + b"\n")
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        finally:
            s.close()
        if b"\n" not in buf:
            raise InferenceError("connection_closed", "broker closed without a reply")
        reply = json.loads(buf.split(b"\n", 1)[0])
        if not reply.get("ok"):
            raise InferenceError(str(reply.get("error", "unknown")), str(reply.get("detail", "")))
        return reply


class LiteHarness:
    """Stdlib mirror of `auspexai_tenant.executor.ExecutorHarness`.

    Usage (inside your vendored executor):

        from lite import LiteHarness  # or auspexai_tenant.lite when authoring

        def run_one(unit, models_dir):
            return {"score": 0.42}

        if __name__ == "__main__":
            import sys
            sys.exit(LiteHarness(run_one).main())
    """

    def __init__(self, run_one) -> None:
        self._run_one = run_one

    def main(self, argv: list | None = None) -> int:
        p = argparse.ArgumentParser(prog="tenant-executor-lite")
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--models", required=True)
        p.add_argument("--timeout", type=int, default=600)
        args = p.parse_args(argv)

        try:
            with open(args.input, encoding="utf-8") as f:
                unit = json.load(f)
        except FileNotFoundError as e:
            self._stderr(f"work-unit input not found: {e}")
            return 2
        except json.JSONDecodeError as e:
            self._stderr(f"work-unit input is not valid JSON: {e}")
            return 2
        if not isinstance(unit, dict):
            self._stderr("work-unit input must be a JSON object")
            return 2
        missing = [k for k in _REQUIRED_UNIT_FIELDS if k not in unit]
        if missing:
            self._stderr(f"work-unit input missing required fields: {missing}")
            return 2

        models_dir = Path(args.models)
        if not models_dir.is_dir():
            self._stderr(f"--models path is not a directory: {models_dir}")
            return 2

        try:
            payload = self._run_one(unit, models_dir)
        except Exception as e:
            self._stderr(f"tenant run_one() raised {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
            return 1
        if not isinstance(payload, dict):
            self._stderr(f"tenant run_one() must return dict, got {type(payload).__name__}")
            return 1

        output = {
            "schema_version": "0.1",
            "unit_id": unit["unit_id"],
            "completed_at": datetime.now(UTC).isoformat(),
            "exit_code": 0,
            "payload": payload,
        }
        out_path = Path(args.output)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            tmp_path.rename(out_path)
        except OSError as e:
            self._stderr(f"failed to write output to {args.output}: {e}")
            return 2
        return 0

    @staticmethod
    def _stderr(msg: str) -> None:
        print(f"executor: {msg}", file=sys.stderr)
