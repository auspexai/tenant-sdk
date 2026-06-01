# synth-geometry — archetype-C synthetic tenant

The **§9 #30–33 genericity-pass forcing function**. Sibling to `synth_tenant`
(the integer doubler), but deliberately *maximally different* so it stresses the
platform's tenant-neutral contracts the way a real interpretability/geometry
tenant would:

| | doubler (`synth_tenant`) | **synth-geometry** |
|---|---|---|
| input | integer in payload | none — the **model** is the subject |
| model access | none | loads a weight matrix, does linear algebra |
| output | exact integer | **floating-point** geometric metric |
| consensus | trivial integer equality | exact-hash quorum via the **#33 determinism contract** |
| cross-unit | none | one global "geometry fingerprint" (#34 P1, tenant-side) |

It estimates a geometric separation statistic (mean absolute cosine similarity
over random row pairs of the weight matrix `W`) by **seeded Monte-Carlo** — a
neutral stand-in for the Ramsauer-Δ / causal-inner-product family.

## Why it passes consensus with no coordinator change

The coordinator's only reducer is exact-hash (`hash_agreement`). Two honest
workers computing a float on different hardware would normally differ in the last
bits and spuriously "disagree". The **determinism contract (#33)** bridges it:

1. the **seed** rides in the (identical-across-replicas) work-unit payload;
2. the executor is a deterministic function of `(W, seed)`;
3. float outputs are **canonical-quantized** (`auspexai_tenant.canonical_quantize`)
   so honest noise below the quantum collapses to one canonical value.

So honest replicas bit-match and reach quorum — on the **unchanged** coordinator.
Tolerance/statistical consensus (#31), for tenants whose signal can't be
quantized, is deferred to Phase 2.

The manifest declares `local_weights_required: true` — the existing §5.8
capability-declaration vocabulary. The *match* (scheduler routing only to workers
that have the weights) is the deferred Phase-2 §5.8 implementation; the
declaration path needs no schema change.

## Run

```bash
python make_model.py                 # write model/W.json (deterministic)
python make_workunits.py             # write work_units.tar.gz (seeded units)

# Proof gate — real executor subprocess + REAL coordinator reducer, no mocks.
# Needs a venv with auspexai_tenant AND auspexai_platform:
#   uv venv /tmp/p1proof && uv pip install --python /tmp/p1proof \
#     -e <repo>/tenant-sdk -e <repo>/platform
/tmp/p1proof/bin/python prove_consensus.py
```

`prove_consensus.py` checks: (1) replicas byte-identical, (2) the real
`hash_agreement_reducer` agrees N=3 on the float metric, (3) a tampered replica
is rejected, (4) the tenant-side cross-unit reduce yields a geometry fingerprint.

## Files

- `make_model.py` — deterministic synthetic weight matrix `W`.
- `executor.py` — seeded geometric estimate + #33 quantization (`run_one`).
- `make_workunits.py` — seeded work-unit tarball.
- `reduce_experiment.py` — experiment-level cross-unit reduce (#34 Phase-1).
- `prove_consensus.py` — the pass/fail proof gate.

See `Documentation/AuspexAI/v0.1.0/{genericity_pass_ratification,archetype_c_tenant_design}.md`.
