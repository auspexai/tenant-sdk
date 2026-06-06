"""AuspexAI Tenant SDK.

Apache-2.0 licensed reference implementation of the AuspexAI tenant-side
published contract. The platform itself (coordinator, worker) is AGPL-3.0; this
SDK is permissively licensed by design (see Documentation/AuspexAI/v0.1.0/
sdk_license_boundary_position.md). Tenant code authoring against this SDK is
not a derivative work of the AGPL platform — the boundary is the published
data + subprocess contract, not the SDK's own license.
"""

from importlib.metadata import version as _v

from auspexai_tenant.attestation import (
    AttestationVerification,
    ResultSetAttestation,
    merkle_root,
    verify_against_results,
    verify_attestation,
)
from auspexai_tenant.client import (
    CoordinatorError,
    TenantClient,
    TransferVerification,
    verify_transfer,
)
from auspexai_tenant.determinism import canonical_quantize, canonicalize_floats
from auspexai_tenant.executor import ExecutorFn, ExecutorHarness
from auspexai_tenant.http_signing import Rfc9421Auth, sign_request
from auspexai_tenant.manifest import (
    ApproverAttestation,
    BuiltinReducer,
    CustomReducer,
    Executor,
    HttpWorkUnitSource,
    Manifest,
    Model,
    Reducer,
    SensitiveContentFlag,
    StaticWorkUnitSource,
    WorkUnitSource,
)
from auspexai_tenant.receipts import (
    QuorumAgreement,
    Receipt,
    ResultHashAnchor,
    TimeWindow,
    decode_cbor,
    encode_cbor,
)
from auspexai_tenant.reducer import ReducerDecision, ReducerFn, ReducerHarness
from auspexai_tenant.signing import (
    DEFAULT_KEY_PATH,
    MaintainerKey,
    ManifestSignature,
    sign_manifest,
    verify_manifest,
)
from auspexai_tenant.upload import (
    UploadResult,
    submit_experiment,
    submit_experiment_from_files,
)
from auspexai_tenant.workunits import (
    ExecutorOutput,
    Result,
    WorkUnit,
    tar_reader,
    tar_writer,
)

# Git-derived (hatch-vcs); read from installed metadata. See version_provenance.md.
__version__ = _v("auspexai-tenant")

__all__ = [
    "DEFAULT_KEY_PATH",
    "ApproverAttestation",
    "AttestationVerification",
    "BuiltinReducer",
    "CoordinatorError",
    "CustomReducer",
    "Executor",
    "ExecutorFn",
    "ExecutorHarness",
    "ExecutorOutput",
    "HttpWorkUnitSource",
    "MaintainerKey",
    "Manifest",
    "ManifestSignature",
    "Model",
    "QuorumAgreement",
    "Receipt",
    "Reducer",
    "ReducerDecision",
    "ReducerFn",
    "ReducerHarness",
    "Result",
    "ResultHashAnchor",
    "ResultSetAttestation",
    "Rfc9421Auth",
    "SensitiveContentFlag",
    "StaticWorkUnitSource",
    "TenantClient",
    "TimeWindow",
    "TransferVerification",
    "UploadResult",
    "WorkUnit",
    "WorkUnitSource",
    "__version__",
    "canonical_quantize",
    "canonicalize_floats",
    "decode_cbor",
    "encode_cbor",
    "merkle_root",
    "sign_manifest",
    "sign_request",
    "submit_experiment",
    "submit_experiment_from_files",
    "tar_reader",
    "tar_writer",
    "verify_against_results",
    "verify_attestation",
    "verify_manifest",
    "verify_transfer",
]
