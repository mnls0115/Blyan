# Repository Guidelines

## Project Structure & Module Organization
- `api/`: FastAPI endpoints and server integration (`api/server.py`).
- `backend/`: core blockchain, learning, p2p, and model logic.
- `frontend/`: static HTML/CSS/JS for the web UI.
- `server/`, `scripts/`: process orchestration, utilities, demos.
- `tests/`: pytest-based suites (e.g., `tests/test_dense_learning_e2e.py`).
- `docker/`, `docker-compose.yml`: GPU node containerization and optional Redis.
- `config/`, `data/`, `logs/`: service configs, local chains/data, runtime logs.

## Build, Test, and Development Commands
- `make install`: install Python dependencies into the active venv.
- `python -m api.server`: run the API locally (uvicorn/ FastAPI).
- `./server.sh start|status|logs api`: manage API and local P2P nodes.
- `pytest tests/ -v`: run tests (recommended). `make test` runs a targeted suite.
- `docker-compose up -d`: start GPU node and optional Redis locally.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent, type hints where practical.
- Names: `snake_case` for functions/modules, `CamelCase` for classes.
- Frontend: keep current formatting; file names kebab-case (e.g., `chat-optimized.js`).
- Docstrings: brief, “why + what”; prefer small, testable functions.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio` (see `TEST_PLAN.md`).
- Naming: files as `tests/test_*.py`; functions `test_*`.
- Markers: use `@pytest.mark.e2e`, `@pytest.mark.gpu` when applicable.
- Coverage: aim >80% for `backend`; run `pytest --cov=backend --cov-report=term`.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped, concise; reference issues (e.g., `api: fix streaming chunk flush (#123)`).
- PRs: include summary, rationale, test plan/commands, and screenshots for UI.
- Requirements: lint/style conformance, tests passing, docs updated (API/ARCH/USER as needed).

## Security & Configuration Tips
- Never commit secrets; base env on `.env.example`/`.env.model`.
- Local dev may set `SKIP_POW=true` and `ENABLE_POL=true` (see `server.sh`).
- Use `logs/` and `/pol/status` to verify healthy API before merging.

## Cautions & Principles
- Root cause first: investigate logs, configs, and execution flow before edits.
- No mocks/placeholders/hardcoding: production-ready code only; use env/config; return real errors and status codes.
- Blockchain-only models: reconstruct weights from chain; never load local/HF. If chain access fails, fail clearly.
- Precision policy: BF16 for student/inference/learning; INT8 only for teacher validation. Keep node computations consistent.
- Dense model (not MoE): use pipeline parallelism; avoid expert routing logic.
- Frontend discipline: use `config.js` for API base, sanitize/validate inputs, no fake UI data or debug noise in production.
- Security: never commit secrets; validate blocks and hashes; don’t expose node keys.
