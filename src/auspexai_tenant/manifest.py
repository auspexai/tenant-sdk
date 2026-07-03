"""Manifest Pydantic models, matching schemas/manifest_v0_1.json.

The manifest is the declarative half of the tenant published contract: it is
pure data (JSON), validated against a published JSON Schema (mirror of the
normative CDDL in cddl/manifest_v0_1.cddl). Tenants submit manifests via the
SDK CLI; the AuspexAI coordinator reads them as data, content-addresses them,
and stores them for cross-reference by work units.

Per Principles §5.14, the schema is content-addressed (manifest_sha256 lives in
each work unit) — manifest mutation after submission requires the worker to
re-accept the new hash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

SensitiveContentFlag = Literal[
    "dual_use",
    "red_team",
    "harmful_output_generation",
]


class Model(BaseModel):
    """A model the experiment requires workers to have locally (BYOM — §5.8).

    `hf_repo` + `hf_filename` are optional acquisition coordinates (M3 lazy
    auto-acquire): when present, an auto-acquire-enabled worker that lacks the
    model may pull this exact file (the pinned quant) from HuggingFace rather
    than refusing the unit. They are self-describing — a provisioned worker
    reads them from its locally-staged manifest. Absent coords ⇒ the model must
    be staged out-of-band (the pre-M3 BYOM behavior). `hf_filename` pins the
    exact GGUF so hash-agreement consensus runs the identical quant across
    replicas (the acquisition-side analog of the manifest_sha256 pin)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    local_weights_required: bool
    hf_repo: str | None = None
    hf_filename: str | None = None
    # v0_2 / §9 #13b: the expected sha256 of the served GGUF weights. When set,
    # the coordinator REJECTS a result whose worker-reported served digest does
    # not match — "the declared model provably ran", which unlocks cross-model
    # PERFORMANCE comparison (§11 amd 2). Omit ⇒ behavioral-only (status quo).
    expected_gguf_sha256: Annotated[str | None, Field(pattern=r"^[a-f0-9]{64}$")] = None


class InferenceDeterminism(BaseModel):
    """v0_2 / M1: the generation policy for a consensus inference run. Two modes,
    keyed on the declared temperature (inference_determinism_scoping_memo §3a):

    - **greedy** — `temperature == 0` (or the block omitted): byte-for-byte
      deterministic decoding. The default; every pre-v0.5 manifest is this.
    - **seeded-sampling** — `temperature > 0` with a **pinned `seed`** (required:
      unseeded sampling is refused — the reproducibility floor, §5). The optional
      sampling knobs `top_p`/`top_k`/`min_p` (v0.5, ratified Q2) are a fixed,
      explicitly-enumerated whitelist; the worker's broker honors the declared
      values per-request and rejects anything beyond them.

    The worker enforces the declaration (never a constant), and when
    `serving_version_pin` is set hard-refuses a unit whose serving stack is
    outside the pin (the refusal is retryable → re-offered to an eligible
    worker). Omit the block ⇒ worker defaults (greedy)."""

    model_config = ConfigDict(extra="forbid")

    temperature: Annotated[float, Field(ge=0)] = 0.0
    seed: int | None = None
    serving_version_pin: str | None = None  # e.g. "ollama/0.17.7"
    hardware_class: str | None = None  # e.g. "cpu" | "cuda"
    # v0_5: the seeded-sampling whitelist (memo Q2). Only meaningful under
    # sampling — declaring a knob at temperature 0 is rejected (an unread
    # declaration is the declarative-enforcement bug class).
    top_p: Annotated[float | None, Field(gt=0, le=1)] = None
    top_k: Annotated[int | None, Field(ge=1)] = None
    min_p: Annotated[float | None, Field(ge=0, lt=1)] = None

    @property
    def is_sampling(self) -> bool:
        return self.temperature > 0

    @property
    def sampling_knobs(self) -> dict[str, float | int]:
        """The declared v0.5 sampling knobs, by name (empty when none declared)."""
        return {k: v for k in ("top_p", "top_k", "min_p") if (v := getattr(self, k)) is not None}

    @model_validator(mode="after")
    def _sampling_coherence(self) -> InferenceDeterminism:
        if self.is_sampling and self.seed is None:
            raise ValueError(
                "seeded sampling (temperature > 0) requires a pinned 'seed' — "
                "unseeded sampling is not accepted (reproducibility floor)"
            )
        if not self.is_sampling and self.sampling_knobs:
            raise ValueError(
                f"sampling knobs {sorted(self.sampling_knobs)} require temperature > 0 "
                "(greedy decoding never reads them — declare them only under sampling)"
            )
        return self


class OutputSchema(BaseModel):
    """v0_2 / M4: the declared measurement type of a unit's result, so the SDK
    loader + dashboard type results instead of guessing. Stevens typology;
    start minimal (level + shape + dtype), grow later."""

    model_config = ConfigDict(extra="forbid")

    measurement_level: Literal["nominal", "ordinal", "interval", "ratio"]
    shape: list[int] | None = None  # [] scalar · [N] vector · [R,C] matrix
    dtype: str | None = None  # e.g. "float32" | "int" | "string"


class ApproverAttestation(BaseModel):
    """Signed attestation from an Approver-pool member (§5.12, §6.5).

    Required when sensitive_content_flags is non-empty.
    """

    model_config = ConfigDict(extra="forbid")

    approver_pubkey: Annotated[str, Field(pattern=r"^ed25519:[A-Za-z0-9+/=]+$")]
    sig: Annotated[str, Field(pattern=r"^[A-Za-z0-9+/=]+$")]


class StaticWorkUnitSource(BaseModel):
    """Work-unit source: tarball of pre-generated unit_<id>.json files.

    Phase 1 default. Strongest separation — tarball hash gives content-address.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["static"]
    tarball_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class HttpWorkUnitSource(BaseModel):
    """Work-unit source: tenant-hosted HTTPS endpoint the coordinator POSTs to."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["http"]
    url: Annotated[str, Field(pattern=r"^https://")]


WorkUnitSource = Annotated[
    StaticWorkUnitSource | HttpWorkUnitSource,
    Field(discriminator="kind"),
]


class Executor(BaseModel):
    """The tenant executor: forked by the worker in the sandbox subprocess (§5.17).

    `command` is the argv to invoke inside the sandbox. The worker passes
    --input/--output/--models/--timeout per the executor invocation convention
    documented in sdk_license_boundary_position.md §6.4.
    """

    model_config = ConfigDict(extra="forbid")

    command: Annotated[list[str], Field(min_length=1)]
    image_sha256: Annotated[str | None, Field(pattern=r"^[a-f0-9]{64}$")] = None
    # Digest over the executor *files* the tenant stages (computed by
    # `compute_package_digest`). When set, the worker re-derives it over the staged
    # package and refuses on mismatch — so a provisioned worker verifies the CODE
    # it runs is what the tenant *signed*, not just that the manifest JSON matches
    # (closes the provenance gap that local manual staging leaves open, and makes
    # coordinator-served fetch a drop-in). Optional + backward-compatible: a
    # manifest without it keeps the Phase-1 "operator is the trust root" behavior.
    package_sha256: Annotated[str | None, Field(pattern=r"^[a-f0-9]{64}$")] = None


class BuiltinReducer(BaseModel):
    """Built-in SHA-256 hash-agreement reducer (`within_cell_exact`).

    Suitable for deterministic outputs (most hash-agreement result shapes fit).
    Requires byte-exact cross-worker reproducibility — opt-in only on a pinned
    "deterministic cell" (serving_version_pin + temp0/seed). On a heterogeneous
    BYO fleet prefer BuiltinToleranceReducer (C7), which version skew does not break.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["builtin_hash_agreement"]


class BuiltinToleranceReducer(BaseModel):
    """C7 `within_cell_tolerance` — the default corroboration model for inference /
    LLM experiments on a heterogeneous fleet.

    Replicas AGREE when their declared features fall within each feature's
    `comparison` envelope (read from `feature_schema`, NOT re-declared here — the
    feature_schema_design.md §5 reconciliation) for at least `replication_floor`
    workers; a divergent worker is an OUTLIER (recorded in the divergence index),
    not a consensus-blocker. That is the C15 fix: under exact@N one skewed worker
    blocks the unit; under tolerance@floor it is one outlier among the agreeing
    floor. The kind starts with 'builtin' so it clears the certification machine
    gate; it emits the `within_cell_tolerance` integrity basis. See
    tolerance_consensus_design.md (RATIFIED 2026-06-29).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["builtin_within_cell_tolerance"]
    # Optional subset of feature_schema paths whose `comparison` forms the
    # agreement predicate. Omit ⇒ every feature that declares a non-`exact`
    # `comparison` (the anchor's `exact` rule strengthens the basis but never
    # blocks — §9 Q3). The coordinator validates each path is a declared feature.
    tolerance_features: Annotated[list[str], Field(min_length=1)] | None = None


class CustomReducer(BaseModel):
    """Tenant-supplied reducer: forked by the coordinator in a subprocess.

    Same boundary mechanism as the executor (subprocess + filesystem IO).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["custom"]
    command: Annotated[list[str], Field(min_length=1)]
    image_sha256: Annotated[str | None, Field(pattern=r"^[a-f0-9]{64}$")] = None


Reducer = Annotated[
    BuiltinReducer | BuiltinToleranceReducer | CustomReducer,
    Field(discriminator="kind"),
]


ResearchClass = Literal[
    "behavioral_drift",
    "eval_sweeps",
    "refusal_boundary_mapping",
    "cross_model_comparison",
    "quantization_effects",
    "prompt_sensitivity",
    "other",
]
"""The §11 research-class taxonomy (§9 #48 keys auto-approval on it). Optional on
the manifest; the coordinator is authoritative — it re-validates membership AND
that the class is within the tenant's approved application classes, then decides
auto-approve vs human review by class x tenant tier."""


# ── D16.1: the self-describing feature schema (v0.3) ─────────────────────────
# Every executor-emitted feature DECLARES what it means, its type + bounds, what a
# change implies, and how replicas are compared for consensus. One declaration,
# several readers (feature_schema_design.md §6): the coordinator enforces the
# type/bounds at result ingest (a §7 structural guarantee — there is no free-text
# kind), the C7 reducer reads `comparison` (the tolerance envelope IS this
# declaration), and load_verified surfaces meaning/unit/range/role so a column
# arrives self-documented. The schema lives in the SIGNED manifest, so the
# documentation is content-addressed, attested, and fixed before any data exists.

FeatureKind = Literal["hash", "count", "numeric", "categorical", "ordinal", "set"]
"""The §7-safe measurement type. There is DELIBERATELY no `text`/free-string kind:
a string feature must be a `categorical` with a CLOSED `categories` set, or a
`hash` — that is the no-raw-text guarantee made structural, not conventional."""

FeatureRole = Literal["anchor", "summary", "provenance", "key", "diagnostic"]
"""Feature AUTHORITY. anchor = the authoritative drift/consensus signal (e.g.
response_sha256); summary = a coarse, possibly lossy aggregate (never the sole
drift truth); provenance = what produced the row (stratify by it); key = a
join/comparison coordinate (feeds D16.2 comparison_keys / D16.4 drift joins);
diagnostic = operational, not a research measure."""

SetElementKind = Literal["categorical", "numeric", "count", "hash"]


class FeatureRange(BaseModel):
    """Closed/half-open numeric bound for `numeric`/`count` features. `max` omitted
    ⇒ unbounded above (a count). Read at result ingest to reject out-of-range
    values (§7) and by load_verified for the data dictionary."""

    model_config = ConfigDict(extra="forbid")

    min: float
    max: float | None = None

    @model_validator(mode="after")
    def _ordered(self) -> FeatureRange:
        if self.max is not None and self.max < self.min:
            raise ValueError(f"range max ({self.max}) < min ({self.min})")
        return self


class ValidWhen(BaseModel):
    """A STRUCTURED validity predicate (never free-text/eval — the M4 lesson, so it
    is a *reader* not dead text): the feature is interpretable only when
    `<field> <op> <value>` holds. load_verified evaluates it → a `<col>.valid`
    column / warning (e.g. type_token_ratio is degenerate when tokens < 5)."""

    model_config = ConfigDict(extra="forbid")

    field: str
    op: Literal[">=", "<=", ">", "<", "==", "!="]
    value: float | int | str


class FeatureComparison(BaseModel):
    """How replicas/rounds must AGREE on this feature — the C7 tolerance envelope,
    declared once here and read by the within_cell_tolerance reducer (C7 Inc 1).
    Rule types mirror tolerance_consensus_design.md §3.1."""

    model_config = ConfigDict(extra="forbid")

    rule: Literal["exact", "numeric", "set_jaccard", "categorical_exact"]
    rel: float | None = None  # numeric: relative tolerance
    abs: float | None = None  # numeric: absolute tolerance
    min: float | None = None  # set_jaccard: minimum similarity

    @model_validator(mode="after")
    def _rule_fields(self) -> FeatureComparison:
        if self.rule == "numeric" and self.rel is None and self.abs is None:
            raise ValueError("comparison rule 'numeric' needs 'rel' or 'abs'")
        if self.rule == "set_jaccard":
            if self.min is None or not (0.0 <= self.min <= 1.0):
                raise ValueError("comparison rule 'set_jaccard' needs 'min' in [0,1]")
        if self.rule in ("exact", "categorical_exact") and (
            self.rel is not None or self.abs is not None or self.min is not None
        ):
            raise ValueError(f"comparison rule '{self.rule}' takes no rel/abs/min")
        return self


class FeatureDeclaration(BaseModel):
    """A self-describing declaration for one executor-emitted feature (D16.1).
    Keyed in `Manifest.feature_schema` by the dotted result-payload path the
    executor emits (e.g. "lexical.type_token_ratio"); load_verified prefixes the
    column with "output.". `kind` is load-bearing — it fixes both the §7 ingest
    validation and the default comparison rule."""

    model_config = ConfigDict(extra="forbid")

    # core (required)
    meaning: Annotated[str, Field(min_length=1)]
    kind: FeatureKind
    role: FeatureRole
    change_means: Annotated[str, Field(min_length=1)]
    # §7-safe bounds (required FOR THE KIND — see the validator)
    unit: str | None = None
    range: FeatureRange | None = None
    categories: list[str] | None = None  # required+non-empty for categorical/ordinal (validator)
    algorithm: str | None = None  # hash, e.g. "sha256"
    element_kind: SetElementKind | None = None  # set
    max_cardinality: Annotated[int | None, Field(ge=1)] = None  # set
    # interpretability (optional)
    invariant_to: list[str] = Field(default_factory=list)
    valid_when: ValidWhen | None = None
    # consensus (optional; defaults per kind in C7)
    comparison: FeatureComparison | None = None

    @model_validator(mode="after")
    def _kind_bounds(self) -> FeatureDeclaration:
        k = self.kind
        # Each kind requires its own §7 bounds...
        if k == "numeric" and self.range is None:
            raise ValueError("kind 'numeric' requires a 'range' (a bounded measure)")
        if k in ("categorical", "ordinal") and not self.categories:
            raise ValueError(
                f"kind '{k}' requires a non-empty 'categories' "
                "(a CLOSED set — the §7 no-free-text guarantee)"
            )
        if k == "hash" and not self.algorithm:
            raise ValueError("kind 'hash' requires an 'algorithm' (e.g. 'sha256')")
        if k == "set" and (self.element_kind is None or self.max_cardinality is None):
            raise ValueError("kind 'set' requires 'element_kind' and 'max_cardinality'")
        # ...and rejects bounds that do not belong to it (no nonsense declarations).
        if self.range is not None and k not in ("numeric", "count"):
            raise ValueError(f"'range' is only valid for numeric/count, not '{k}'")
        if self.categories is not None and k not in ("categorical", "ordinal", "set"):
            raise ValueError(f"'categories' is only valid for categorical/ordinal/set, not '{k}'")
        if self.algorithm is not None and k != "hash":
            raise ValueError(f"'algorithm' is only valid for hash, not '{k}'")
        if (self.element_kind is not None or self.max_cardinality is not None) and k != "set":
            raise ValueError(f"'element_kind'/'max_cardinality' are only valid for set, not '{k}'")
        return self


class PreRegistration(BaseModel):
    """D16.2: the pre-registered experiment design (preregistration_design.md §3).

    Declares the hypothesis, analysis, and decision rule BEFORE any data exists.
    The tenant signature makes the block tamper-evident; the coordinator's
    submit-time Rekor anchor makes `design ≺ data` publicly provable (the anchor
    at submit precedes the result attestation's anchor at completion). Required
    for a citable/DOI'd result (§7 — a TECHNICAL gate, ratified no-waiver);
    optional for exploratory runs.

    The comparison envelope is NOT re-declared here: `features` NAMES declared
    feature_schema paths, and the pre-registered envelope IS each feature's
    `comparison` in the same signed manifest — one declaration, nothing to drift
    (the D16.1 §5 reconciliation precedent; the design doc's own "declare the
    same thing once"). Every field describes the DESIGN over declared features —
    §7-safe, never raw output. A post-hoc analysis change is a signed, append-only
    deviation record, never an edit (§5)."""

    model_config = ConfigDict(extra="forbid")

    # Q2 (ratified as proposed): the block is SDK-versioned like the result schema.
    version: Literal["0.1"] = "0.1"
    hypothesis: Annotated[str, Field(min_length=20, max_length=2000)]
    analysis_method: Annotated[str, Field(min_length=10, max_length=2000)]
    # Dotted feature_schema paths the analysis reads — each must be declared and
    # must carry a `comparison` (the pre-registered envelope). Manifest-validated.
    features: Annotated[list[str], Field(min_length=1)]
    timescale: Literal["intra_experiment_rounds", "long_horizon", "inter_experiment"]
    decision_rule: Annotated[str, Field(min_length=10, max_length=2000)]
    expected_result: Annotated[str, Field(min_length=5, max_length=2000)]
    # Q4: required — a declared stopping rule precludes "ran until it looked
    # good" (data-peeking-dependent stopping is visible as a deviation).
    stopping_rule: Annotated[str, Field(min_length=10, max_length=1000)]
    # Stable join keys for the comparison (esp. inter-experiment drift series).
    comparison_keys: list[str] = Field(default_factory=list)
    # Q4: optional power declaration (minimum corroboration for the claim).
    min_replication: Annotated[int | None, Field(ge=1)] = None


class Manifest(BaseModel):
    """AuspexAI tenant manifest, v0.1 through v0.5.

    Mirrors schemas/manifest_v0_1.json … v0_5.json. Each version is a superset
    of the prior, enforcement keyed on PRESENCE: v0.2 adds four optional members
    (M1-M4); v0.3 adds the optional `feature_schema` (D16.1); v0.4 adds the
    optional `pre_registration` (D16.2); v0.5 extends `inference_determinism`
    with the seeded-sampling whitelist (top_p/top_k/min_p — memo Q2). Older
    versions stay valid forever (re-verify-forever, no forced migration). See
    sdk_license_boundary_position.md §6.2 for the published-contract framing.
    """

    model_config = ConfigDict(extra="forbid")

    # Each version stays valid forever (re-verify-forever; no forced migration).
    # Every superset member is OPTIONAL — enforcement keys on PRESENCE, so a
    # manifest declaring none of a version's members is structurally the prior
    # version with schema_version bumped.
    schema_version: Literal["0.1", "0.2", "0.3", "0.4", "0.5"]
    tenant_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")]
    tenant_maintainer_contact: EmailStr
    experiment_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{2,127}$")]
    research_goal_paragraph: Annotated[str, Field(min_length=50, max_length=2000)]
    # §9 #48: the declared research class. Optional for back-compat (a manifest
    # without it routes to human review — the agent can't auto-clear an
    # unclassified experiment).
    research_class: ResearchClass | None = None
    models: Annotated[list[Model], Field(min_length=1)]
    prompt_set_characteristics: Annotated[str, Field(min_length=10, max_length=1000)]
    sensitive_content_flags: list[SensitiveContentFlag] = Field(default_factory=list)
    approver_attestations: list[ApproverAttestation] | None = None
    expected_duration_hours: Annotated[float, Field(gt=0, le=8760)]
    replication_factor: Annotated[int, Field(ge=1, le=100)]
    work_unit_source: WorkUnitSource
    executor: Executor
    reducer: Reducer
    # v0_2 members (all optional; enforcement keys on presence):
    # M2 — promote the field the coordinator already derives informally to a
    # declared one: a real-execution experiment with no model requirement still
    # must be kept off synthetic/echo workers.
    requires_real_execution: bool = False
    # M1 — the determinism profile (worker pins temp/seed, hard-refuses outside
    # the serving-version pin).
    inference_determinism: InferenceDeterminism | None = None
    # M4 — the declared measurement type of a unit's result.
    output_schema: OutputSchema | None = None
    # v0_3 / D16.1 — the self-describing feature schema, keyed by the dotted
    # result-payload path the executor emits. Optional (enforcement keys on
    # presence); required for certified/citable experiments, optional for BYOT
    # (the coordinator gate, Inc 2). See feature_schema_design.md.
    feature_schema: dict[str, FeatureDeclaration] | None = None
    # v0_4 / D16.2 — the pre-registered design. Optional (exploratory runs);
    # required for a citable/DOI'd result. See preregistration_design.md.
    pre_registration: PreRegistration | None = None

    @model_validator(mode="after")
    def _sampling_coherent_with_reducer(self) -> Manifest:
        """The §3c coherence gate, mirrored at build (the D16.2 precedent:
        enforced here AND at coordinator submit). Sampled replicas legitimately
        differ, so an agreement reducer would either spuriously fail or falsely
        claim corroboration — sampling requires a non-agreement collection mode
        (process-only / a distributional fold via a custom reducer)."""
        det = self.inference_determinism
        if (
            det is not None
            and det.is_sampling
            and self.reducer.kind in ("builtin_hash_agreement", "builtin_within_cell_tolerance")
        ):
            raise ValueError(
                f"seeded sampling (temperature > 0) is incoherent with the agreement "
                f"reducer {self.reducer.kind!r} — sampled replicas legitimately differ; "
                "declare a non-agreement collection mode (inference_determinism memo §3c)"
            )
        return self

    @model_validator(mode="after")
    def _sampling_knobs_require_v0_5(self) -> Manifest:
        """The top_p/top_k/min_p whitelist is a v0.5 contract extension — the
        published v0.2-v0.4 schema artifacts are immutable and do not carry it."""
        det = self.inference_determinism
        if det is not None and det.sampling_knobs and self.schema_version != "0.5":
            raise ValueError(
                f"sampling knobs {sorted(det.sampling_knobs)} require schema_version "
                '"0.5" (the published v0.2-v0.4 schemas are immutable and do not '
                "carry them)"
            )
        return self

    @model_validator(mode="after")
    def _sensitive_requires_attestation(self) -> Manifest:
        if self.sensitive_content_flags and not self.approver_attestations:
            raise ValueError(
                "approver_attestations is required when sensitive_content_flags is non-empty "
                "(Principles §5.12 research ethics)"
            )
        return self

    @model_validator(mode="after")
    def _pre_registration_references_declared_features(self) -> Manifest:
        """D16.2: a pre-registration must be CHECKABLE against this same manifest —
        its analysis features exist in the feature_schema and each carries the
        `comparison` envelope the design pre-registers (referenced, never
        duplicated); its comparison_keys are declared features too. This is the
        epistemic-integrity seam D16.5 guards: enforced at build AND mirrored at
        coordinator submit."""
        pr = self.pre_registration
        if pr is None:
            return self
        if self.schema_version in ("0.1", "0.2", "0.3"):
            raise ValueError(
                'a manifest declaring pre_registration must set schema_version "0.4" '
                "or later (the D16.2 published-contract member)"
            )
        fs = self.feature_schema or {}
        if not fs:
            raise ValueError(
                "pre_registration requires a feature_schema (the design is declared "
                "over self-describing features — D16.1)"
            )
        for path in pr.features:
            decl = fs.get(path)
            if decl is None:
                raise ValueError(f"pre_registration feature {path!r} is not in feature_schema")
            if decl.comparison is None:
                raise ValueError(
                    f"pre_registration feature {path!r} declares no 'comparison' — "
                    "the pre-registered envelope IS the feature's comparison; declare it there"
                )
        for key in pr.comparison_keys:
            if key not in fs:
                raise ValueError(
                    f"pre_registration comparison_key {key!r} is not in feature_schema"
                )
        return self


# Files never part of the executor package digest: the manifest itself (separately
# content-addressed) and Python bytecode caches (runtime pollution).
# manifest.json.sig is excluded because `manifest sign` writes it NEXT TO the
# manifest by default (i.e. into the package dir) AFTER the digest was computed
# — including it would poison executor.package_sha256 for any tenant staging
# their package dir wholesale. The worker reimplementation excludes it
# identically (the contract is the format, not shared code).
_PACKAGE_DIGEST_EXCLUDE = ("manifest.json", "manifest.json.sig")


def _iter_package_files(package_dir: str | Path) -> Iterator[tuple[str, Path]]:
    """Yield `(posix-relpath, path)` for every file that is part of the executor
    package, sorted by relpath, applying the digest exclusions.

    The single enumeration seam shared by `compute_package_digest` and the
    archive builder (`auspexai_tenant.package.build_package_archive`) — using
    one walk guarantees the uploaded archive and the `X-Package-Digest` header
    can never disagree about which files a package contains.
    """
    root = Path(package_dir)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        if rel in _PACKAGE_DIGEST_EXCLUDE or rel.endswith(".pyc") or "__pycache__" in parts:
            continue
        yield rel, path


def compute_package_digest(package_dir: str | Path) -> str:
    """Digest over the executor *files* staged in `package_dir`, for the manifest's
    `executor.package_sha256` (the provenance pin the worker verifies).

    Canonical + deterministic: every regular file under `package_dir`, except
    `manifest.json` / `manifest.json.sig` and any `__pycache__/` / `*.pyc`,
    contributes one line
    ``<posix-relpath>\\x00<sha256-hex>`` (lines sorted by relpath, joined by
    ``\\n``); the digest is the SHA-256 of that blob. The worker re-implements this
    byte-for-byte (`auspexai_worker.provisioning.compute_package_digest`) — the
    shared contract is the format, not shared code (SDK is Apache, worker AGPL).
    """
    lines: list[str] = []
    for rel, path in _iter_package_files(package_dir):
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{rel}\x00{file_hash}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
