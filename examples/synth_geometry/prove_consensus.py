#!/usr/bin/env python3
"""Proof: archetype-C runs on the UNCHANGED coordinator consensus path.

This is the genericity-pass forcing function's pass/fail gate. It uses the
**real executor subprocess** (auspexai_tenant) and the **real coordinator
agreement function** (``auspexai_platform.receipts.issuance.hash_agreement_reducer``)
— no mocks — to show that a float-output, weights-only archetype-C tenant, under
the #33 determinism contract, reaches exact-hash quorum exactly as the integer
doubler does, with zero coordinator change.

Run from a venv that has BOTH ``auspexai_tenant`` and ``auspexai_platform``:

    uv pip install --python <platform-venv> -e ../tenant-sdk     # add SDK to it
    <platform-venv>/bin/python prove_consensus.py

Checks:
  1. each unit's 3 replicas produce byte-identical canonical {exit_code,payload};
  2. the real ``hash_agreement_reducer`` returns agreed=True, agreeing_workers=3;
  3. negative control: perturbing one replica's float -> agreed=False;
  4. the tenant-side cross-unit reduce (#34 P1) yields a geometry fingerprint.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import make_model  # noqa: E402
from auspexai_platform.db.models import Result  # noqa: E402
from auspexai_platform.receipts.issuance import hash_agreement_reducer  # noqa: E402
from reduce_experiment import reduce_metrics  # noqa: E402

from auspexai_tenant.workunits import WorkUnit  # noqa: E402

EXECUTOR = HERE / "executor.py"
PUBKEYS = [f"{i:064x}" for i in (0xA1, 0xB2, 0xC3)]
N_UNITS = 6
N_SAMPLES = 1500
REPLICAS = 3

_failures = 0


def _check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    mark = "PASS" if ok else "FAIL"
    if not ok:
        _failures += 1
    print(f"[{mark}] {label}" + (f": {detail}" if detail else ""))


def _run_replica(unit_path: Path, models_dir: Path, out_path: Path) -> dict:
    subprocess.run(
        [sys.executable, str(EXECUTOR),
         "--input", str(unit_path), "--output", str(out_path),
         "--models", str(models_dir), "--timeout", "60"],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out_path.read_text())


def _to_result(env: dict, idx: int) -> Result:
    return Result(
        result_id=f"res-{env['unit_id']}-{idx}",
        unit_id=env["unit_id"],
        worker_id=f"wkr-{idx}",
        worker_pubkey_hex=PUBKEYS[idx],
        exit_code=env["exit_code"],
        payload=env["payload"],
        worker_signature="ZGVtby1zaWc=",
        completed_at=datetime.now(UTC),
        received_at=datetime.now(UTC),
    )


def main() -> int:
    print("=== synth-geometry: archetype-C consensus proof ===")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models = tmp / "models"
        models.mkdir()
        (models / "W.json").write_text(
            json.dumps(make_model.build_matrix(rows=128, cols=32, seed=20260601),
                       separators=(",", ":"), sort_keys=True)
        )

        per_unit_metric: list[float] = []
        first_unit_results: list[Result] = []
        now = datetime.now(UTC)

        for u in range(1, N_UNITS + 1):
            unit = WorkUnit(
                schema_version="0.1", unit_id=f"g{u:04d}",
                tenant_id="synth-geometry", experiment_id="synth-geometry-v1",
                manifest_sha256="0" * 64, created_at=now,
                payload={"seed": 1000 + u, "n_samples": N_SAMPLES,
                         "probe": "mean_abs_cosine_separation"},
            )
            unit_path = tmp / f"{unit.unit_id}.json"
            unit_path.write_text(unit.model_dump_json())

            envelopes = []
            for r in range(REPLICAS):
                out_path = tmp / f"{unit.unit_id}-r{r}.json"
                envelopes.append(_run_replica(unit_path, models, out_path))

            # (1) replicas byte-identical on the hashed {exit_code, payload}.
            canon = {
                json.dumps({"exit_code": e["exit_code"], "payload": e["payload"]},
                           sort_keys=True, separators=(",", ":"))
                for e in envelopes
            }
            _check(f"{unit.unit_id}: {REPLICAS} replicas byte-identical",
                   len(canon) == 1,
                   detail=f"metric={envelopes[0]['payload']['mean_abs_cosine_separation']}")

            # (2) the REAL coordinator reducer agrees.
            results = [_to_result(e, i) for i, e in enumerate(envelopes)]
            outcome = hash_agreement_reducer(results)
            _check(f"{unit.unit_id}: hash_agreement_reducer agreed (N={REPLICAS})",
                   outcome.agreed and outcome.agreeing_workers == REPLICAS,
                   detail=f"semantic_hash={(outcome.semantic_hash or '')[:16]}…")

            per_unit_metric.append(
                float(envelopes[0]["payload"]["mean_abs_cosine_separation"]))
            if not first_unit_results:
                first_unit_results = results

        # (3) negative control: perturb one replica's float -> disagreement.
        tampered = list(first_unit_results)
        bad_payload = dict(tampered[0].payload)
        bad_payload["mean_abs_cosine_separation"] += 0.0001
        tampered[0] = tampered[0].model_copy(update={"payload": bad_payload})
        neg = hash_agreement_reducer(tampered)
        _check("negative control: tampered replica -> NOT agreed", not neg.agreed)

        # (4) tenant-side cross-unit reduce (#34 P1).
        fingerprint = reduce_metrics(per_unit_metric)
        _check("experiment-level reduce produced a fingerprint",
               fingerprint["unit_count"] == N_UNITS)
        print("geometry fingerprint:", json.dumps(fingerprint))

    print(f"=== {'ALL CHECKS PASSED' if _failures == 0 else f'{_failures} CHECK(S) FAILED'} ===")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
