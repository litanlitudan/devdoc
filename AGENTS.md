# Repository Guidelines

## Project Structure & Module Organization
- `lib/` TypeScript source (edit here). `dist/` is compiled output (do not edit).
- `bin/` CLI entry points (`markserv`, `readme`).
- `tests/` unit tests and `tests/fixtures/` sample content.
- `scripts/` helper Bash/Python (MLIR/ONNX tooling, build helpers).
- `media/`, `lib/templates/`, `lib/icons/` static assets and templates.

## Build, Test, and Development Commands
- `npm run build` — compile TS to `dist/` and run post-build.
- `npm start` — run Markserv on port `8642` from `dist/`.
- `npm run dev` — kill port, build once, start with Node watch.
- `npm test` — build then run Vitest; `npm run test:watch` / `test:ui` for dev.
- `npm run cover` — run tests with coverage.
- `npm run lint` / `npm run format` — ESLint fix, Prettier write.
- `npm run port:kill` — free port `8642` if something is bound.
- Make targets: `make dev`, `make test`, `make cover`, `make setup-python`.

## Coding Style & Naming Conventions
- Language: TypeScript (ES2022). Keep runtime code in `lib/**/*.ts`.
- Formatting: Prettier (tabs, width 2, single quotes, no semicolons).
- Linting: ESLint (flat config in `eslint.config.js`) with Prettier.
- Files: use kebab-case for files, camelCase for variables, PascalCase for classes.
- Do not commit changes to `dist/`; build artifacts are generated.

## Testing Guidelines
- Framework: Vitest. Place tests in `tests/unit/*.test.js`.
- Use fixtures from `tests/fixtures/` for sample inputs; avoid network.
- Prefer small, focused tests; assert HTML/text output deterministically.
- Run `npm run cover` before PR; include new tests for new behavior/bugs.

## Commit & Pull Request Guidelines
- Commits: imperative present (“Add X”, “Fix Y”), concise subject + details.
- Reference issues with `#123` in body when applicable.
- PRs: include a clear description, screenshots for UI/rendering changes, and steps to verify.
- Keep PRs scoped; update docs in `README.md`/`GLOBAL_USAGE.md` when user-facing changes.

## Security & Configuration Tips
- Requires Node `>=18`. Default HTTP port: `8642`.
- Optional: `npm run setup:python` for ONNX/MLIR features.
- Avoid editing files under `third_party/` unless explicitly needed.

## Agent-Specific Notes
- Edit source under `lib/`; never hand-edit `dist/`.
- Follow this guide for all subdirectories of the repo.
