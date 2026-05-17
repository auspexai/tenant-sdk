"""AuspexAI Tenant SDK.

Apache-2.0 licensed reference implementation of the AuspexAI tenant-side
published contract. The platform itself (coordinator, worker) is AGPL-3.0; this
SDK is permissively licensed by design (see Documentation/AuspexAI/v0.1.0/
sdk_license_boundary_position.md). Tenant code authoring against this SDK is
not a derivative work of the AGPL platform — the boundary is the published
data + subprocess contract, not the SDK's own license.
"""

from auspexai_tenant.executor import ExecutorFn, ExecutorHarness
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
from auspexai_tenant.upload import UploadResult, upload_manifest
from auspexai_tenant.workunits import (
    ExecutorOutput,
    Result,
    WorkUnit,
    tar_reader,
    tar_writer,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_KEY_PATH",
    "ApproverAttestation",
    "BuiltinReducer",
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
    "SensitiveContentFlag",
    "StaticWorkUnitSource",
    "TimeWindow",
    "UploadResult",
    "WorkUnit",
    "WorkUnitSource",
    "__version__",
    "decode_cbor",
    "encode_cbor",
    "sign_manifest",
    "tar_reader",
    "tar_writer",
    "upload_manifest",
    "verify_manifest",
]
