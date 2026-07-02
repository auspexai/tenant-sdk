"""D16.5 — epistemic-integrity conformance, SDK side (pass #8 Push 2).

The platform twin (auspexai-platform tests/test_epistemic_conformance.py) pins
the coordinator mirror to the published v0.4 contract via a literal; THIS suite
pins the SDK's Pydantic model to the published schema FILE. Change the contract
→ both suites fail together, by design (the AUD-22 named-reader discipline:
no epistemic field can exist that only one side reads).
"""

from __future__ import annotations

from auspexai_tenant.manifest import PreRegistration
from auspexai_tenant.schemas import load_schema


def test_invariant_model_matches_published_schema() -> None:
    """INVARIANT: the PreRegistration model's field set equals the published
    v0.4 pre_registration properties — required and optional alike."""
    schema = load_schema("manifest_v0_4.json")
    block = schema["properties"]["pre_registration"]
    model_fields = set(PreRegistration.model_fields)
    assert model_fields == set(block["properties"])
    required = {name for name, f in PreRegistration.model_fields.items() if f.is_required()}
    assert required == set(block["required"])


def test_invariant_contract_carries_no_second_envelope() -> None:
    """INVARIANT (one-declaration): the published pre_registration block has no
    comparison-bearing member — the envelope lives ONLY in feature_schema,
    referenced by feature name. A `comparison` property appearing here would
    reopen the three-way drift seam D16.2 deliberately closed."""
    schema = load_schema("manifest_v0_4.json")
    block = schema["properties"]["pre_registration"]
    assert "comparison" not in block["properties"]
    assert block["additionalProperties"] is False  # nothing can sneak in
