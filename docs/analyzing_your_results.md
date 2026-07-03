# Analyzing your results — a researcher's field guide

This is the companion to [`reading_your_evidence.md`](reading_your_evidence.md). That guide answers **"can I trust this number?"** (verification, corroboration, the apparatus footprint). This one answers the next three questions:

1. **How do I analyze the data?** (the workflow + recipes)
2. **What tools do I use?**
3. **How do I interpret the data?** (what every column *means*)

The running example throughout is the certified **Vigiles drift probe** — the experiment AuspexAI ships and fully documents, so you have a worked reference to copy.

> **The golden rule still applies: verify, *then* analyze.** Everything below assumes you've verified the bundle (see `reading_your_evidence.md`). `load_verified` enforces it for you — it raises rather than hand you unverified data.

---

## Part 0 — the 60-second path

```bash
# 1. download + verify the evidence bundle (custody transfer; verify is on by default)
auspexai-tenant experiment export <exp-id> -o evidence.json

# 2. read what every column MEANS, straight from the signed manifest
auspexai-tenant bundle table evidence.json --data-dictionary

# 3a. get a flat table for Excel / R / Tableau …
auspexai-tenant bundle table evidence.json -o results.csv      # or results.parquet
```
```python
# 3b. … or a verified pandas DataFrame, in one call
from auspexai_tenant import evidence
df = evidence.load_verified("evidence.json")   # raises unless the bundle verifies
```

That's the whole loop: **export → verify → understand the columns → analyze.** The rest of this guide is detail.

---

## Part 1 — tools: what you actually need

You do **not** need a heavy install for most of this. The SDK is split so the core stays light (it's also installed on workers, drivers, and CI that never touch a DataFrame).

| You want to… | You need | Command / call |
|---|---|---|
| Download + verify a bundle | **base SDK** (`pip install auspexai-tenant`) | `auspexai-tenant experiment export` / `bundle verify` |
| Read the **data dictionary** | **base SDK** (no pandas) | `auspexai-tenant bundle table … --data-dictionary` |
| Read the raw values yourself | nothing — it's JSON | open `evidence.json` in any language |
| A **pandas DataFrame** (`load_verified`) | the **`[analysis]`** extra | `pip install 'auspexai-tenant[analysis]'` |
| A **CSV / Parquet** table | the **`[analysis]`** extra | `auspexai-tenant bundle table … -o results.csv` |

The `[analysis]` extra is just **`pandas` + `pyarrow`** — the only heavy dependencies, needed only for the DataFrame/table convenience. Verification and the data dictionary never need them.

**Bring your own tools.** The frame is ordinary pandas, and `bundle table -o` writes plain CSV/Parquet — so from here you're in **Excel, Google Sheets, R (`read.csv`/`arrow`), Tableau, DuckDB, Julia**, whatever you already use. AuspexAI doesn't impose an analysis environment; it hands you verified, self-documented data and gets out of the way.

---

## Part 2 — how to analyze

### Step 1 — get the columns, then *read what they mean*

Before you touch a value, print the data dictionary. It is generated from the **signed** manifest, so it's the attested documentation — not a wiki someone forgot to update:

```bash
auspexai-tenant bundle table evidence.json --data-dictionary
```
```
Data dictionary — 11 declared feature(s), from the signed manifest:

  output.response_sha256  [hash/anchor]  (sha256)
      SHA-256 of the raw model output — the authoritative byte-level drift anchor
      Δ a change means: ANY change = output bytes differed, incl. reordering/whitespace the lexical features cannot see
  output.lexical.type_token_ratio  [numeric/summary]  (ratio · [0.0, 1.0])
      unique tokens / total tokens — lexical diversity
      Δ a change means: vocabulary richness shifted; does NOT capture reordering or formatting drift
  …
```

This is the single biggest defense against guesswork: under AuspexAI's containment (§7) **raw model output is never retained** — the declared features are the *entire* interpretability surface. If a feature isn't in the dictionary, it doesn't exist in your data; if it is, the dictionary tells you its meaning, unit, bounds, role, and *what a change in it implies.*

### Step 2 — load the verified frame

```python
from auspexai_tenant import evidence
df = evidence.load_verified("evidence.json")
```
One row per consensus result. Alongside your `output.*` feature columns and `input.*` work-unit columns, the frame carries:

| Column / attr | What it is | Source |
|---|---|---|
| `output.<feature>` | your declared features | the result payload |
| `output.<feature>.valid` | **validity flag** — `False` where the feature is degenerate (see Part 3) | `valid_when` in the schema |
| `df.attrs["feature_schema"]` | the full data dictionary, keyed by column | the signed manifest |
| `integrity_basis` | corroboration class of the row (`within_cell_exact` / `within_cell_tolerance` / `process_only` / `diverged`) | the attestation |
| `served_weights`, `output.model.gguf_sha256` | provenance — which model produced the row | worker-attested |
| `df.attrs["governance_footprint"]` | the apparatus footprint | the attestation |

(The trust columns — `integrity_basis`, `ran_under`, the footprint — are covered in depth in `reading_your_evidence.md`. **Stratify by them; don't pool.**)

### Step 3 — the analysis recipes

> **Confirmatory vs exploratory (pre-registration).** If the run was
> pre-registered (the certified Vigiles starter is — manifest v0.4's
> `pre_registration` block), its `analysis_method`, `decision_rule`, and
> `stopping_rule` were declared in the signed manifest and anchored in the
> public transparency log *before any data existed* (`design ≺ data` — the
> trust guide covers the verify lines). Analysis inside that declaration is
> **confirmatory**; the recipes below are fair game as **exploratory**
> follow-ups — just report them as such, and if you adopt a changed analysis,
> declare it: `auspexai-tenant experiment deviate <exp> --what "…" --why "…"`
> (append-only + signed; zero deviations means the pre-registered analysis
> stands, and that claim is machine-checked).


**Recipe A — drop degenerate rows (always do this first).**
```python
# a feature is only interpretable where its .valid flag isn't False
ttr = df[df["output.lexical.type_token_ratio.valid"] != False]
```

**Recipe B — stratify by provenance (compare like with like).**
```python
# never pool rows produced by different served weights unless cross-model IS the question
for digest, g in df.groupby("output.model.gguf_sha256"):
    ...   # analyze each model build separately
```

**Recipe C — the Vigiles drift analysis (per-probe stability across rounds).**
The Vigiles research question is: *does a fixed probe's output stay stable across rounds?* The **anchor** (`response_sha256`) is the truth; a changed anchor = drift.
```python
for probe, g in df.groupby("output.probe_id"):
    n = g["output.response_sha256"].nunique()
    print(probe, "→", "STABLE" if n == 1 else f"DRIFTED ({n} distinct outputs)")
```

**Recipe D — inter-experiment drift series (longitudinal).**
To track drift *across* runs (e.g. weekly), export each run and join on the **key** features (`probe_id`, seed), stratifying by provenance (`gguf_sha256`):
```python
frames = [evidence.load_verified(f).assign(run=f) for f in ["wk1.json", "wk2.json", "wk3.json"]]
series = pd.concat(frames)
pivot = series.pivot_table(index="output.probe_id", columns="run",
                           values="output.response_sha256", aggfunc="first")
# a row whose hash changes column-to-column drifted between runs
```
*(A first-class longitudinal surface for this is planned — D16.4. The recipe above is the manual version.)*

**No-code path.** `bundle table -o results.csv`, then pivot in Excel/Sheets, or `read.csv` in R. The `--data-dictionary` output is your column legend.

---

## Part 3 — how to interpret: the feature schema is your key

Every feature declares five things that remove the guesswork. Read them off `df.attrs["feature_schema"]["output.<col>"]` or the `--data-dictionary` output.

### `role` — the interpretation discipline (read this first)

`role` tells you **how much authority a feature has**. Mixing them up is the most common analysis error:

| role | meaning | how to treat it |
|---|---|---|
| **anchor** | the authoritative truth (`response_sha256`) | this is what "changed / didn't change" means. Drift = the anchor moved. |
| **summary** | a coarse, lossy aggregate (`lexical.*`) | **never the sole truth.** A summary can miss changes the anchor catches — check its `invariant_to`. |
| **key** | a join / stratify coordinate (`probe_id`, seed) | group and compare *by* these |
| **provenance** | what produced the row (`model.id`, `model.gguf_sha256`) | stratify or exclude by these |
| **diagnostic** | operational, not a research measure | don't put it in your results |

### `change_means`, `unit`, `range` — what a movement implies
Each feature spells out what a change *means* (`change_means`), its `unit`, and its valid `range`. You never have to infer whether a number going up is good, expected, or out of bounds.

### `valid_when` + the `.valid` column — when a feature lies
Some features are only interpretable under a precondition. `lexical.type_token_ratio` (unique/total tokens) is **mathematically 1.0 whenever there's one token** — which looks like *maximal* diversity but is the opposite. So it declares `valid_when: tokens ≥ 5`, and `load_verified` adds an `output.lexical.type_token_ratio.valid` column. **Filter on it.** (In the shipped Vigiles run, 10 of 15 rows had `tokens < 5` and would have been misread without this flag.)

### `invariant_to` — a summary's declared blind spots
A summary feature lists what it *cannot see*. `lexical.type_token_ratio` is `invariant_to: [token_order, whitespace, punctuation]` — so if the model reorders or re-spaces its output, the TTR won't budge but **the anchor's hash will.** That's not a contradiction; it's the anchor doing its job.

### The three cautionary tales (real findings from the shipped run)
These are why the role discipline matters — all observed in the certified Vigiles run:

1. **The anchor caught what every summary missed.** On one round, `p-instruction`'s `response_sha256` changed while *every lexical feature stayed identical* — the output was reordered/re-spaced, invisible to the order-invariant summaries. **Lesson: trust the anchor for "did it change."**
2. **A summary that "looked perfect" was degenerate.** `p-refusal` collapsed to a single token, so `type_token_ratio = 1.0` — "maximal diversity," actually a one-word answer. The `.valid` flag catches it. **Lesson: respect `valid_when`.**
3. **Longest ≠ most stable.** The longest output (`p-greeting`, 146 chars) was byte-identical across all rounds; the shortest, most-structured output drifted. **Lesson: drift is about the anchor, not length** — don't proxy stability with a `summary` like `response_chars`.

### The Vigiles features, fully interpreted (the reference to copy)

| column | kind / role | read it as |
|---|---|---|
| `output.response_sha256` | hash / **anchor** | the drift truth — any change = the bytes differed |
| `output.probe_id` | categorical / **key** | which probe; group/compare by this |
| `output.model.id`, `output.model.gguf_sha256` | categorical/hash / **provenance** | which model+weights; stratify/exclude by this |
| `output.lexical.type_token_ratio` | numeric / summary | lexical diversity — **only when `.valid`**; blind to order/whitespace |
| `output.lexical.top_tokens` | set / summary | most-frequent tokens (≤8); blind to order |
| `output.lexical.tokens` / `unique_tokens` | count / summary | token counts; the TTR denominator |
| `output.response_chars`, `output.eval_count` | count / summary | length signals — **not** drift signals |
| `output.schema` | categorical / provenance | the result schema version |

---

## Quick reference

```bash
auspexai-tenant experiment export <exp-id> -o evidence.json   # download + verify (custody transfer)
auspexai-tenant bundle verify evidence.json [--check-rekor]   # re-verify any time
auspexai-tenant bundle table evidence.json --data-dictionary  # what every column means (no pandas)
auspexai-tenant bundle table evidence.json -o results.csv     # flat table → your tools  [needs analysis extra]
```
```python
from auspexai_tenant import evidence
df = evidence.load_verified("evidence.json")          # verified frame  [needs analysis extra]
df.attrs["feature_schema"]["output.<col>"]            # the column's meaning/role/change_means
df[df["output.<col>.valid"] != False]                 # interpretable rows only
```

**See also:** [`reading_your_evidence.md`](reading_your_evidence.md) — verification, corroboration (`integrity_basis`), the apparatus footprint, and citing the Rekor anchor. Read both: that guide tells you *how much to trust* each row; this one tells you *what each row means.*

## The Drift Benchmark — one comparable number (envelope units)

*(The ratified standard for communicating drift — `drift_benchmark_design.md`.)*

Your features are deliberately orthogonal, so no single raw metric is "the
drift number." The benchmark makes them comparable by dividing each declared
scalar comparison's delta by **its own calibrated envelope** (the same
`comparison` your manifest declares and consensus enforces):

- **1.0 envelope units (EU)** = the calibrated boundary between same-behavior
  noise and drift. Below 1 is noise; above 1 the behavior moved.
- **Headline = peak + breadth**: the worst probe's EU, plus the fraction of
  probes beyond 1 EU (one drifted probe reads differently from a panel-wide
  shift). When only one number fits, quote the peak.
- **Byte divergence is reported separately, never folded in** — byte-different
  outputs can be behaviorally identical (and features can be blind where the
  byte anchor is not).
- **Always name your reference**: a score is *against* a specific reference
  experiment, whose signed manifest defines the envelope in force.

```bash
auspexai-tenant benchmark drift runs/<label>/bundle.json runs/<reference>/bundle.json
```

Both inputs are verified evidence bundles (custody + attestation checks run
first). Calibration anchors for intuition, from production data: same
config re-run ≈ 0 EU · seeded sampling (temp 0.8) ≈ 6.7 EU on the open-ended
probe only · a different model ≈ 10 EU panel-wide.


### Non-consensus evidence in the benchmark (D19)

Observe-only extra replicas (`integrity_basis == "observation"`) join the
drift-benchmark scoring set by default — an observe-only run at replication N
scores all N observations. Diverged and outlier payloads never move the
headline scalar: they appear in your DataFrame and in the divergence overlay,
and `benchmark drift --include-diverged` (or
`drift_benchmark_bundles(..., include_diverged=True)`) scores them explicitly
when you are doing apparatus forensics — e.g. attributing drift to a serving-
engine release. The headline number always means *custody-verified,
integrity-passing behavior*.
