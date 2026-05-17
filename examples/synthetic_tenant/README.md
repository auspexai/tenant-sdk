# Synthetic test tenant — integer doubler

The §5.3 forcing function from `AuspexAI_Principles_and_Scope.md`: a deliberately
non-LLM-shaped tenant that exercises every AuspexAI SDK contract. Its purpose
is to keep the SDK honest — to expose any abstractions that secretly assumed
Sentinel-shaped (multi-agent LLM) workloads.

If the SDK can host this synthetic tenant alongside Sentinel, it's general.

## What it does

Each work unit contains a single integer. The executor doubles it. The
reducer (built-in hash-agreement) decides whether the workers agreed.

There is no model. There are no prompts. There is no temperature, no token
count, no agent, no persona, no behavioral drift. The point is to test the
SDK's *shape* without testing anything Sentinel-specific.

## File layout

```
synthetic_tenant/
├── README.md           # this file
├── manifest.json       # tenant manifest (validated against schemas/manifest_v0_1)
├── executor.py         # the doubling executor (uses ExecutorHarness)
└── make_workunits.py   # generates a tarball of unit_<n>.json files
```

The placeholder model entry in `manifest.json` reflects a known schema constraint
(manifest requires `models` with `min_length: 1` — see `schemas/manifest_v0_1.json`).
A future minor-version bump may allow `models: []` for compute-only experiments;
until then, synthetic tenants declare a `local_weights_required: false` placeholder.

## Running locally

```sh
# 1. Generate work-unit tarball
uv run python examples/synthetic_tenant/make_workunits.py

# 2. Validate the manifest (should print OK)
uv run auspexai-tenant manifest validate examples/synthetic_tenant/manifest.json

# 3. Generate maintainer key + sign manifest
uv run auspexai-tenant key generate --output examples/synthetic_tenant/key.pem
uv run auspexai-tenant manifest sign \
    examples/synthetic_tenant/manifest.json \
    --key examples/synthetic_tenant/key.pem

# 4. Run the executor against one work unit directly (mimics what the worker does)
mkdir -p /tmp/synth/{input,output,models}
cat > /tmp/synth/input/unit.json <<'EOF'
{"schema_version":"0.1","unit_id":"u1","tenant_id":"synth-doubler","experiment_id":"synth-doubler-v1","manifest_sha256":"0000000000000000000000000000000000000000000000000000000000000000","created_at":"2026-05-17T20:00:00Z","payload":{"value":21}}
EOF
uv run python examples/synthetic_tenant/executor.py \
    --input /tmp/synth/input/unit.json \
    --output /tmp/synth/output/result.json \
    --models /tmp/synth/models
cat /tmp/synth/output/result.json  # expect payload.doubled == 42
```

## How the test suite uses it

The integration tests at `tests/test_synthetic_tenant.py` invoke
`executor.py` as a subprocess against a crafted work unit and verify the
output structure. This is how the SDK validates that:

- The published wire-format contracts (manifest, workunit, executor-output) are
  implementable by non-LLM tenants
- The ExecutorHarness API is general enough to write a tenant in under 30 LOC
- The platform's tenant-neutral data model (§5.1) is genuinely tenant-neutral
