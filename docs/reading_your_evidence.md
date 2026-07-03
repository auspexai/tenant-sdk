# Reading your evidence — a researcher's guide

When you take custody of an experiment's evidence bundle it carries far more than result values: each result is wrapped in context that tells you *how much to trust it* and *how to combine it*. This guide turns that context into analysis. (If a decision ever goes against you, see your recourse in [`TENANT_TERMS.md` §8](https://github.com/auspexai/.github/blob/main/TENANT_TERMS.md).)

> **This guide is about trust** — *can I trust this number?* For **what each column means and how to analyze it** (the data dictionary, the role discipline, the analysis recipes, the tools), see its companion: [`analyzing_your_results.md`](analyzing_your_results.md). Read both.

## The one rule: verify, *then* analyze

Never analyze raw bundle data. Run verification first — it is the line between "numbers someone sent me" and "numbers I can defend":

```bash
# download the evidence bundle AND verify the whole chain
auspexai-tenant experiment export <exp-id> --verify -o evidence.json

# confirm the attestation is anchored in the public transparency log (Rekor)
auspexai-tenant experiment attestation <exp-id> --check-rekor

# or re-verify a saved bundle offline, any time
auspexai-tenant bundle verify evidence.json
```

or in code — the one call that gates everything:

```python
from auspexai_tenant import evidence

df = evidence.load_verified("evidence.json")   # raises if anything fails to verify
```

`verify_bundle` (run by all of the above) checks: the proof-of-transfer signature, external key pinning, the attestation block + Merkle-root unification, **completeness** (no rows dropped), **input binding** (each result ties to its work unit), **per-result worker signatures**, **tolerance evidence** (each tolerance-consensus unit's attested representative hash recomputes from the bundled evidence), and — for a pre-registered run — the **pre-registration binding** plus **`design ≺ data`** ordering and **deviation disclosure** (see the section below). **A DataFrame in hand is already trustworthy data** — if it didn't verify, you don't have it.

## Where the context lives (two layers — use both)

**1. Per-result columns** — in the `load_verified` DataFrame, one row per consensus result:

| Column | What it tells you | Use it to |
|---|---|---|
| `integrity_basis` | what the result's assurance *rests on*: `within_cell_exact` (≥2 replicas byte-identical) · `within_cell_tolerance` (≥2 replicas agreed **within each feature's declared, calibrated `comparison` envelope** — the attested consensus value is a deterministic *representative* of the agreeing set, and the bundle's `unit_consensus` block carries the per-feature spread + outlier count; this is what the dashboard's "N workers agreed within tolerance · M outliers" line renders) · `process_only` (no cross-worker corroboration claimed: either one worker ran it, or — under the `builtin_process_only` OBSERVE-ONLY reducer, v0.6 — every replica is a deliberate independent observation; the result row's consensus evidence says which, and under observe-only differing outputs are the design, not a fault) · `diverged` (replicas disagreed) | **stratify, don't pool** — a `process_only` row and a corroborated row are different evidence classes |
| `ran_under` | the sandbox the worker **signed** it ran under (`strict` / `permissive`) | stratify by containment; treat `permissive` as lower-assurance |
| `served_weights` | the worker-attested model digest `{model_id: gguf_sha256}` that produced the row | confirm the model; compare only across matching digests unless cross-model IS the question |
| `semantic_hash` | the canonical content hash the consensus agreed on | dedupe / join |
| `aged_off` | the payload was retention-aged-off (the receipt + hash survive) | exclude from value analysis; provenance still verifies |
| `input.*` / `output.*` | your flattened work-unit input + result payload | your actual analysis columns |

**2. The apparatus footprint** (firewall #2) — one per experiment, on the attestation's `governance_footprint` (the same block the dashboard's Integrity panel renders). These are **not** per-result columns; read them off the footprint:

| Footprint field | What it tells you | Use it to |
|---|---|---|
| independence (`distinct_accounts` / `distinct_workers` / `distinct_served_models`) | how independent the consensus producers were | down-weight agreement from non-independent producers (firewall #3) |
| replication + `integrity_basis` counts | how heavily the experiment replicated, and the agree/diverge mix | weight by N; read the network divergence rate |
| containment (`required` vs `ran_under`) | what the apparatus required vs what ran | cross-check against the per-result `ran_under` |
| approval path (auto vs human) + tenant tier | the apparatus's own gating of this experiment | provenance context |
| `generation` (`mode` + `params`) | the generation policy the signed manifest declared: `greedy` (byte-deterministic decoding — replicas *should* match) vs `seeded_sampling` (declared temperature + pinned seed + whitelist knobs — replicas *legitimately differ*; each is an independent sample) | interpret divergence **in kind**: under `seeded_sampling`, differing outputs are the design, not a fault; under `greedy`, they are a signal |

**3. The transparency anchor** — the attestation's Rekor inclusion (`logIndex` + `integratedTime`): a public log entry letting a third party confirm the attestation existed at a point in time, independent of you or AuspexAI.

## A best-practice recipe

1. **Verify first.** No verify, no analysis.
2. **Stratify, don't pool.** Split before you aggregate — pooling different bases or sandboxes silently mixes evidence classes:
   ```python
   df = evidence.load_verified("evidence.json")
   # analyze each integrity class separately, and report them separately
   for basis, g in df.groupby("integrity_basis"):
       print(basis, len(g), g["output.score"].mean())
   # and never average a strict row together with a permissive one
   strict_only = df[df.ran_under == "strict"]
   ```
3. **Weight by replication + independence.** Carry N (replication) and the footprint's independence into your error bars, not just the point estimate. A `process_only` single-worker row is not a replicated one.
4. **Correct for the apparatus footprint.** Treat it as a covariate to subtract, not noise to ignore — that is firewall #2's whole purpose: *"the network watched, and here is how much that mattered."*
5. **Treat divergence as signal, not failure.** Under firewall #1 a worker that disagrees with quorum earns a *divergence receipt* and **equal trust** — it is never discarded. A `diverged` row that is otherwise fully attested is often the most interesting datapoint:
   ```python
   diverged = df[df.integrity_basis == "diverged"]   # look HERE — don't drop these
   ```
   And a *low* network divergence rate (the footprint's `integrity_basis` counts) can mean genuine agreement — or an apparatus that discouraged disagreement. The equal-trust model exists so the rate reflects the former; it is worth checking.
6. **Cite the anchor.** When a claim must be externally defensible, cite the Rekor entry so anyone can confirm the attestation existed without trusting you or us:
   > Attested in the public transparency log (Rekor) at **logIndex 1770786010**, integratedTime **2026-06-09T17:42:31Z** — independently checkable at the configured Rekor instance.

## Pre-registration & deviations — was the claim declared before the data?

A pre-registered run (manifest v0.4's `pre_registration` block — the certified Vigiles starter ships one) carries the strongest epistemic claim a bundle can make: **the hypothesis, analysis, and stopping rule were fixed and publicly anchored *before* any result existed.** Three verify lines cover it:

- **`pre-reg: ok`** — the submit-time anchor statement verifies and binds *this* bundle's manifest; the design in your manifest is provably the one that was anchored (analysis-matches-declared).
- **`design<data: ok`** — the design's transparency-log anchor **precedes** the result attestation's (`design ≺ data`). It reads *pending* until the hourly Rekor sweep has anchored both — pending is never a failure; **FAIL means the design was anchored after the results** (retroactive pre-registration), and the bundle correctly refuses to verify.
- **`deviations: N declared`** — the append-only history of signed analysis changes. **Zero deviations means the pre-registered analysis stands** — and that claim is machine-checked. Declaring deviations never fails a bundle (honest exploratory work is the point; `auspexai-tenant experiment deviate <exp> --what … --why …`); a deviation record that fails to *verify* (a post-hoc edit) does.

Report accordingly: an analysis in the pre-registered block is **confirmatory**; anything else is **exploratory** — cite the deviation record that discloses it. The result attestation's signed predicate also carries the pre-registration reference, so the finding and its design are cryptographically bound.

## What this protects you from

- **Over-claiming** from unreplicated or low-independence results.
- **Mixing evidence classes** (strict + permissive, or different `integrity_basis`) into one misleading average.
- **Mistaking apparatus influence for a finding.**
- **Analyzing tampered or incomplete data** — verification + completeness catch it before you ever see a frame.
- **Post-hoc analysis masquerading as confirmatory** — `design ≺ data` + deviation disclosure make the exploratory/confirmatory line checkable, not a matter of trust.
