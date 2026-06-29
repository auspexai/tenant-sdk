# AuspexAI Tenant SDK

[![CI](https://github.com/auspexai/tenant-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/auspexai/tenant-sdk/actions/workflows/ci.yml)

The researcher surface of the [AuspexAI](https://auspexai.network) volunteer
compute network: everything a research tenant needs to **author** an
experiment, **submit** it to the live coordinator, **drive** it to
convergence, and walk away holding a **cryptographically verified copy of the
data** — receipts, a Rekor-anchored attestation, and a signed custody record
included.

Python ≥ 3.11, Apache-2.0. Install from PyPI:

```bash
pip install auspexai-tenant
# with the analysis surface (pandas/Parquet evidence loader):
pip install "auspexai-tenant[analysis]"
```

Most researchers don't install the SDK directly — the
[researcher onramp](https://getresearcher.auspexai.network) (`curl -fsSL
https://getresearcher.auspexai.network | bash`) provisions it alongside the
dashboard. Releases are also on the
[releases page](https://github.com/auspexai/tenant-sdk/releases),
Sigstore-signed.

**New researcher?** The full front door — install, apply for a tenant via
`auspexai-tenant apply` (GitHub device flow, no key material to paste), and
your first verified run — is [ONBOARDING.md](https://github.com/auspexai/.github/blob/main/ONBOARDING.md).

## The researcher lifecycle

The SDK covers the whole arc. Each stage works today against the live
coordinator (`https://coord.auspexai.network`).

### 1. Author

- **Published wire contracts** (JSON Schema + CDDL, immutable per the
  honor-forever versioning policy) for manifests, work units, executor
  output, results, reducer decisions, and receipts — with strict Pydantic
  models for all of them. Tenants may consume the SDK or implement the
  published contract independently.
- **`ExecutorHarness`** — wrap your science in a
  `run_one(unit, models_dir) -> dict` function and the harness handles the
  worker invocation contract (a working tenant is ~30 lines; see
  [`examples/synthetic_tenant/`](examples/synthetic_tenant/)).
- **`ReducerHarness`** — same pattern for custom reduction.
- **`auspexai_tenant/lite.py`** — a zero-dependency, stdlib-only,
  Python 3.10-safe mirror of the harness, designed to be **vendored into
  your package** so your executor runs in the worker sandbox without
  installing the SDK there. Includes **`InferenceClient`** for
  LLM-inference tenants: deterministic chat against the model the worker
  serves over a sandbox-local socket (temperature pinned to 0 by the worker;
  seed/num_predict/num_ctx whitelisted).
- Static work-unit packing (`tar_writer`/`tar_reader`) and package-digest
  helpers that match the worker's verification byte-for-byte.

### 2. Sign & submit

- Ed25519 tenant keypair (`key generate` / `key pubkey`); every API call is
  signed (RFC 9421 HTTP message signatures) — no passwords, no tokens.
- `manifest validate` → `manifest sign` → `manifest upload`. Manifests are
  immutable once submitted (manifest-swap protection: work units bind the
  manifest hash); a maintainer approves the experiment before any work is
  scheduled.
- **Model catalog**: `model catalog` surfaces the network's bottom-up model
  catalog — the emergent set of models the active workers declare they serve.

### 3. Drive

- **`run_until`** — the autonomic driver: submit a round, fold results into
  a checkpointable aggregate (`Counter`/`Mean`/`Histogram` or your own) as
  they arrive, test your convergence condition, generate the next batch.
  Resumable via a round-atomic journal; wakes on an SSE doorbell instead of
  hot-polling; pluggable stall policy (abort / finalize-partial / keep
  waiting).
- CLI: `experiment run <id> --driver yourmodule:build` drives a whole
  adaptive experiment headlessly. `experiment reduce` batch-reduces a
  completed consensus set.

### 4. Watch

- `experiment list` / `status` / `results` (consensus by default, `--raw`
  for all replicas) / `receipts`, plus an anonymized
  activity rollup (contributor counts, replication fill — never volunteer
  identities).

### 5. Verify — the part you can't get from a rented GPU

Every completed experiment yields a **result-set attestation**: a Merkle
root over the per-unit consensus set, COSE-signed by the network's published
key and anchored in the public
[Sigstore Rekor](https://rekor.sigstore.dev) transparency log. The SDK
verifies all of it independently:

- `experiment attestation --check-rekor --verify-against-results` —
  recompute the root from your own pulled results, check the signature
  against the published signer roster, and confirm public log inclusion.
- v1 attestations additionally bind each **work unit's input hash** into the
  leaves: *"result R was produced from parameters P"* is cryptography, not a
  database row.
- Per-result **worker signatures** — the only signatures in the chain not
  made by the coordinator — verify from the bundle alone.

### 6. Take custody & analyze

- `experiment export --verify` downloads the **evidence bundle** — results +
  work-unit inputs + receipts + attestation + Rekor proof + a signed
  proof-of-transfer — and runs the full verification chain on receipt.
  Collection transfers data custody to you: the network can re-verify your
  bundle forever, but after age-off it never re-delivers. Your verified copy
  is the durable copy.
- `bundle verify <file>` — re-verify a saved bundle any time, offline, with
  no coordinator. Hand the file and the command to a reviewer.
- **`evidence.load_verified(bundle)`** — run the chain, then get a tidy
  pandas DataFrame (inputs as `input.*` columns, outputs as `output.*`).
  There is deliberately no force flag: analysis begins from a verified
  dataset or it doesn't begin. `bundle table <file> -o results.parquet`
  is the no-Python path to Excel/Tableau/R.
  See [`examples/evidence_loader.ipynb`](examples/evidence_loader.ipynb).
- **Two researcher field guides — read both:**
  - **[Reading your evidence](docs/reading_your_evidence.md)** (*trust* — can I trust this number?): each verified row carries `integrity_basis`, `ran_under` (containment), and `served_weights`, and the experiment carries an apparatus footprint. **Stratify, don't pool**; weight by replication + independence; treat divergence as signal, not failure; correct for the footprint.
  - **[Analyzing your results](docs/analyzing_your_results.md)** (*interpretation* — what does it mean, how do I analyze it?): the data dictionary (`bundle table --data-dictionary`), the `role` discipline (anchor vs summary), the `.valid` flag, the analysis recipes (stratify-by-provenance, per-probe drift, inter-experiment drift-series), and which tools to use. Vigiles is the worked exemplar.

## CLI quick reference

| Group | What it does |
|---|---|
| `key` | Ed25519 tenant keypair (generate, pubkey) |
| `manifest` | validate, sign, upload an experiment manifest |
| `experiment` | list, status, run (autonomic driver), results, receipts, attestation, reduce, export |
| `bundle` | verify a saved evidence bundle offline; write it as a CSV/Parquet table |
| `model` | browse the network model catalog |
| `receipts` | decode/pretty-print CBOR receipts |

Every command has `--help` with the full contract.

## Examples

- [`examples/synthetic_tenant/`](examples/synthetic_tenant/) — the minimal
  tenant (integer doubler): manifest, executor, reducer in ~30 lines.
- [`examples/synth_geometry/`](examples/synth_geometry/) — a real numeric
  workload with an adaptive `run_until` driver.
- [`examples/evidence_loader.ipynb`](examples/evidence_loader.ipynb) —
  export → verify → DataFrame → Parquet.

## Status

**Open beta.** The coordinator is live and experiments run on real volunteer
hardware end-to-end (work-unit dispatch, consensus, receipts, live Rekor
anchoring, evidence-bundle custody). The network is publicly installable:
researchers self-onboard with the
[getresearcher](https://getresearcher.auspexai.network) one-liner and apply for
a tenant via `auspexai-tenant apply` (GitHub or ORCID device flow — no
invitation, no key material to paste). A maintainer reviews the application
against the [Research Ethics
Policy](https://github.com/auspexai/.github/blob/main/RESEARCH_ETHICS_POLICY.md);
from there a **certified starter** runs with no code review — the safest first
run — while bringing your own code is reviewed more closely. Questions about
running research on the network? Email **contact@auspexai.network**.

## Scope & license boundary

The Tenant SDK is the contract between tenant research code and the AuspexAI
platform. It is intentionally a thin client over a **published data +
subprocess contract**: tenants consume the SDK's stable surface (or
implement the published contract independently); the AGPL-3.0 platform
(`auspexai/platform`, `auspexai/worker`) runs on the other side of that
contract. Tenant project modules authoring against the SDK are not
derivative works of the AGPL platform — the split follows the
published-contract pattern used by MongoDB drivers, Grafana plugins, and
Sentry SDKs (copyleft on the server, permissive on the SDK). See the
AuspexAI Principles & Scope §5.2 for the structural argument.

[Apache-2.0](LICENSE) for the SDK. Tenants license their own research code
however suits them.

## Governance & policies

- [Governance](https://github.com/auspexai/.github/blob/main/GOVERNANCE.md) — roles, decision rules, recruitment, conflict of interest
- [Code of Conduct](https://github.com/auspexai/.github/blob/main/CODE_OF_CONDUCT.md) — community standards, reporting, escalation pathway
- [Contributing](https://github.com/auspexai/.github/blob/main/CONTRIBUTING.md) — DCO sign-off, PR workflow, RFC requirement for substantial changes
- [Research Ethics Policy](https://github.com/auspexai/.github/blob/main/RESEARCH_ETHICS_POLICY.md) — what research can run on the network and how it's reviewed
- [Tenant Terms](https://github.com/auspexai/.github/blob/main/TENANT_TERMS.md) — including the results-custody model this SDK implements
