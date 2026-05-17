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

The SDK is intentionally a thin client over a published data + subprocess contract. Tenants consume the SDK's stable surface (or implement the published contract independently); the AGPL-3.0 platform (`auspexai/platform`, `auspexai/worker`, `auspexai/coordinator`) runs on the other side of that contract. Tenant project modules authoring against the SDK are not derivative works of the AGPL platform — see the AuspexAI Principles & Scope §5.2 and `Documentation/AuspexAI/v0.1.0/sdk_license_boundary_position.md` for the structural argument and the comparator survey (MongoDB, Grafana, Nextcloud, Mastodon).

## License

[Apache-2.0](LICENSE) for the SDK. Tenants are free to license their own tenant project code under whatever license suits their research — Sentinel ships under Apache-2.0; future tenants choose their own.

The AuspexAI **platform** (separate repos: `auspexai/platform`, `auspexai/worker`, `auspexai/coordinator`) is AGPL-3.0. The SDK/platform license split follows the published-contract pattern used by MongoDB drivers, Grafana plugins, and Sentry SDKs — copyleft on the server, permissive on the SDK.

## Governance & policies

- [Governance](https://github.com/auspexai/.github/blob/main/GOVERNANCE.md) — roles, decision rules, recruitment, conflict of interest
- [Code of Conduct](https://github.com/auspexai/.github/blob/main/CODE_OF_CONDUCT.md) — community standards, reporting, escalation pathway
- [Contributing](https://github.com/auspexai/.github/blob/main/CONTRIBUTING.md) — DCO sign-off, PR workflow, RFC requirement for substantial architectural changes
- [Research Ethics Policy](https://github.com/auspexai/.github/blob/main/RESEARCH_ETHICS_POLICY.md) — what AI safety research can run on the network and how it's reviewed

## Watch this repo

Activity will begin as Phase 1 ramps up.
