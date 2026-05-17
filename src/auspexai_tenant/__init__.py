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
from auspexai_tenant.reducer import ReducerDecision, ReducerFn, ReducerHarness
from auspexai_tenant.signing import (
    DEFAULT_KEY_PATH,
    MaintainerKey,
    ManifestSignature,
    sign_manifest,
    verify_manifest,
)
from auspexai_tenant.upload import UploadResult, upload_manifest
from auspexai_tenant.workunits import ExecutorOutput, Result, WorkUnit

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
    "Reducer",
    "ReducerDecision",
    "ReducerFn",
    "ReducerHarness",
    "Result",
    "SensitiveContentFlag",
    "StaticWorkUnitSource",
    "UploadResult",
    "WorkUnit",
    "WorkUnitSource",
    "__version__",
    "sign_manifest",
    "upload_manifest",
    "verify_manifest",
]
