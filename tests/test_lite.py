"""auspexai_tenant.lite — the vendorable stdlib executor kit (W-S step 4).

The lite harness must mirror the ExecutorHarness contract (CLI, exit codes,
output shape — cross-validated against the real pydantic ExecutorOutput),
and the InferenceClient must speak the worker broker's line-delimited JSON
protocol (exercised against a real unix-socket fake broker).
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from auspexai_tenant.lite import InferenceClient, InferenceError, LiteHarness
from auspexai_tenant.workunits import ExecutorOutput


def _unit_dict(**overrides) -> dict:
    unit = {
        "schema_version": "0.1",
        "unit_id": "u-1",
        "tenant_id": "t-1",
        "experiment_id": "exp-label",
        "manifest_sha256": "ab" * 32,
        "created_at": "2026-06-09T12:00:00+00:00",
        "payload": {"x": 1},
    }
    unit.update(overrides)
    return unit


def _write_input(tmp_path: Path, unit: dict | str) -> Path:
    p = tmp_path / "input.json"
    p.write_text(unit if isinstance(unit, str) else json.dumps(unit), encoding="utf-8")
    return p


def _argv(tmp_path: Path, input_path: Path) -> list[str]:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    return [
        "--input",
        str(input_path),
        "--output",
        str(tmp_path / "out.json"),
        "--models",
        str(models),
        "--timeout",
        "60",
    ]


class TestLiteHarness:
    def test_happy_path_output_matches_sdk_contract(self, tmp_path: Path):
        seen = {}

        def run_one(unit, models_dir):
            seen["unit"] = unit
            seen["models_dir"] = models_dir
            return {"score": 0.42}

        rc = LiteHarness(run_one).main(_argv(tmp_path, _write_input(tmp_path, _unit_dict())))
        assert rc == 0
        assert seen["unit"]["unit_id"] == "u-1"
        assert isinstance(seen["models_dir"], Path)

        raw = json.loads((tmp_path / "out.json").read_text())
        # Cross-contract: the lite output validates as the REAL SDK
        # ExecutorOutput (pydantic) — the worker-side reader's shape.
        out = ExecutorOutput.model_validate(raw)
        assert out.unit_id == "u-1"
        assert out.exit_code == 0
        assert out.payload == {"score": 0.42}
        assert not (tmp_path / "out.json.tmp").exists()  # atomic rename

    def test_tolerates_additive_unknown_fields(self, tmp_path: Path):
        unit = _unit_dict(some_future_field="ok")
        rc = LiteHarness(lambda u, m: {"ok": True}).main(
            _argv(tmp_path, _write_input(tmp_path, unit))
        )
        assert rc == 0

    def test_missing_required_field_is_harness_failure(self, tmp_path: Path):
        unit = _unit_dict()
        del unit["manifest_sha256"]
        rc = LiteHarness(lambda u, m: {}).main(_argv(tmp_path, _write_input(tmp_path, unit)))
        assert rc == 2

    def test_malformed_json_is_harness_failure(self, tmp_path: Path):
        rc = LiteHarness(lambda u, m: {}).main(
            _argv(tmp_path, _write_input(tmp_path, "not json {"))
        )
        assert rc == 2

    def test_missing_input_is_harness_failure(self, tmp_path: Path):
        rc = LiteHarness(lambda u, m: {}).main(_argv(tmp_path, tmp_path / "absent.json"))
        assert rc == 2

    def test_missing_models_dir_is_harness_failure(self, tmp_path: Path):
        argv = _argv(tmp_path, _write_input(tmp_path, _unit_dict()))
        argv[argv.index("--models") + 1] = str(tmp_path / "no-such-dir")
        rc = LiteHarness(lambda u, m: {}).main(argv)
        assert rc == 2

    def test_tenant_exception_is_tenant_failure(self, tmp_path: Path):
        def run_one(unit, models_dir):
            raise RuntimeError("tenant code broke")

        rc = LiteHarness(run_one).main(_argv(tmp_path, _write_input(tmp_path, _unit_dict())))
        assert rc == 1

    def test_non_dict_return_is_tenant_failure(self, tmp_path: Path):
        rc = LiteHarness(lambda u, m: "nope").main(
            _argv(tmp_path, _write_input(tmp_path, _unit_dict()))
        )
        assert rc == 1


class _FakeBroker:
    """A unix-socket fake speaking the worker broker's wire protocol."""

    def __init__(self, socket_path: Path, replies: dict[str, dict]) -> None:
        self.socket_path = socket_path
        self.requests: list[dict] = []
        self._replies = replies
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(socket_path))
        self._listener.listen(2)
        self._listener.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn:
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                if b"\n" not in buf:
                    continue
                request = json.loads(buf.split(b"\n", 1)[0])
                self.requests.append(request)
                reply = self._replies[request["op"]]
                conn.sendall(json.dumps(reply).encode() + b"\n")

    def close(self) -> None:
        self._stop.set()
        self._listener.close()
        self._thread.join(timeout=2.0)


class TestInferenceClient:
    def test_generate_and_info(self, tmp_path: Path, monkeypatch):
        sock = tmp_path / "inference.sock"
        broker = _FakeBroker(
            sock,
            replies={
                "generate": {
                    "ok": True,
                    "message": {"role": "assistant", "content": "deterministic"},
                    "eval_count": 5,
                    "model": "tiny-q4",
                },
                "info": {
                    "ok": True,
                    "model": "tiny-q4",
                    "gguf_sha256": "ef" * 32,
                    "backend_handle": "auspex-tiny-q4",
                },
            },
        )
        try:
            monkeypatch.setenv("AUSPEXAI_INFERENCE_SOCKET", str(sock))
            monkeypatch.setenv("AUSPEXAI_INFERENCE_MODEL", "tiny-q4")
            client = InferenceClient.from_env()

            reply = client.generate([{"role": "user", "content": "hi"}], options={"seed": 0})
            assert reply["message"]["content"] == "deterministic"
            # The model came from env; options rode along.
            assert broker.requests[0]["model"] == "tiny-q4"
            assert broker.requests[0]["options"] == {"seed": 0}

            info = client.info()
            assert info["gguf_sha256"] == "ef" * 32
        finally:
            broker.close()

    def test_broker_error_raises_typed(self, tmp_path: Path):
        sock = tmp_path / "inference.sock"
        broker = _FakeBroker(
            sock,
            replies={
                "generate": {
                    "ok": False,
                    "error": "params_rejected",
                    "detail": "temperature must be 0",
                }
            },
        )
        try:
            client = InferenceClient(str(sock), model="tiny-q4")
            with pytest.raises(InferenceError) as exc_info:
                client.generate([{"role": "user", "content": "hi"}])
            assert exc_info.value.code == "params_rejected"
        finally:
            broker.close()

    def test_from_env_without_socket_raises(self, monkeypatch):
        monkeypatch.delenv("AUSPEXAI_INFERENCE_SOCKET", raising=False)
        with pytest.raises(RuntimeError, match="not running on an inference-enabled"):
            InferenceClient.from_env()
