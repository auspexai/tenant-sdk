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
from auspexai_tenant.workunits import ExecutorOutput, Result, WorkUnit

__version__ = "0.1.0"

__all__ = [
    "ApproverAttestation",
    "BuiltinReducer",
    "CustomReducer",
    "Executor",
    "ExecutorFn",
    "ExecutorHarness",
    "ExecutorOutput",
    "HttpWorkUnitSource",
    "Manifest",
    "Model",
    "Reducer",
    "Result",
    "SensitiveContentFlag",
    "StaticWorkUnitSource",
    "WorkUnit",
    "WorkUnitSource",
    "__version__",
]
