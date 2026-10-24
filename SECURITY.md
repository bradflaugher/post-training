# Security Policy

POST-TRAINING runs 100% in the browser with no backend server, no accounts, and no telemetry.

## Reporting a Vulnerability

Please report vulnerabilities privately through [GitHub Security Advisories](https://github.com/bradflaugher/post-training/security/advisories/new).

## Safeguards in Place

- All SLM inference and custom prompt probing happen locally in WebAssembly (`env.allowRemoteModels = false`).
- Dynamic DOM updates use `textContent` and `document.createElement` rather than `innerHTML`.
- Strict security headers (`frame-ancestors 'none'`, `Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`, HSTS) are served on every response via `_headers`.
- All GitHub Actions are pinned to full commit SHAs with least-privilege permissions.
