# AuspexAI Tenant SDK

The developer surface for authoring research tenants that run on the [AuspexAI](https://github.com/auspexai) network.

## Status

**Phase 0 — Foundation.** SDK design and code begin in Phase 1, alongside the [platform](https://github.com/auspexai/platform).

## Scope

The Tenant SDK is the contract between tenant project modules (research code) and the AuspexAI Platform. It defines:

- How a tenant declares experiments (job manifests)
- How a tenant ships project code that workers execute
- How a tenant defines its result schema and analysis pipeline
- How a tenant integrates with platform storage, scheduling, and trust layers

The SDK is also the AGPL/non-AGPL **license boundary**. The platform core ships under AGPL-3.0; tenant project modules consuming the SDK's stable surface are not themselves required to be AGPL-3.0. Tenant license is the Researcher's choice — see [`CONTRIBUTING.md`](https://github.com/auspexai/.github/blob/main/CONTRIBUTING.md) Path 2 and the AuspexAI Principles & Scope §5.2 for the boundary design constraint. Counsel review in Phase 1 will validate; until then, treat the AGPL non-infection of tenant code as a working assumption rather than a final guarantee.

## License

[AGPL-3.0](LICENSE) for the SDK itself. Tenant project code authored against this SDK uses whatever license the tenant chooses, subject to the boundary validation noted above.

## Contributing

See [`CONTRIBUTING.md`](https://github.com/auspexai/.github/blob/main/CONTRIBUTING.md) (org-wide). The Tenant SDK is a stable surface; backwards-incompatible changes go through a deprecation cycle and require RFC.

## Governance

Project direction is held by the Maintainer team per [`GOVERNANCE.md`](https://github.com/auspexai/.github/blob/main/GOVERNANCE.md). Code of Conduct: [`CODE_OF_CONDUCT.md`](https://github.com/auspexai/.github/blob/main/CODE_OF_CONDUCT.md).

## Watch this repo

Activity will begin as Phase 1 ramps up.
