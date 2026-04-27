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

## Governance & policies

- [Governance](https://github.com/auspexai/.github/blob/main/GOVERNANCE.md) — roles, decision rules, recruitment, conflict of interest
- [Code of Conduct](https://github.com/auspexai/.github/blob/main/CODE_OF_CONDUCT.md) — community standards, reporting, escalation pathway
- [Contributing](https://github.com/auspexai/.github/blob/main/CONTRIBUTING.md) — DCO sign-off, PR workflow, RFC requirement for substantial architectural changes
- [Research Ethics Policy](https://github.com/auspexai/.github/blob/main/RESEARCH_ETHICS_POLICY.md) — what AI safety research can run on the network and how it's reviewed

## Watch this repo

Activity will begin as Phase 1 ramps up.
