# Contributing to POST-TRAINING

POST-TRAINING is a single-page static web game with no build step and no runtime npm dependencies.

## Setup

```bash
npm ci
pip install -r scripts/requirements.txt -r scripts/requirements-dev.txt
npm run serve
```

## Before opening a PR

```bash
npm run lint
npm run test:unit
npm test
```

See [`AGENTS.md`](AGENTS.md) for architecture details and [`SECURITY.md`](SECURITY.md) for reporting vulnerabilities.
