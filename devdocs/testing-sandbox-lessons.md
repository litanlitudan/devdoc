# Testing in Sandbox Environments — Lessons Learned

## Why the Tests Became Complicated
- **Legacy assumptions**: The original `fetch`-based suites spun up the full HTTP server and hit real `http://localhost:<port>` URLs.
- **Sandbox constraints**: The Codex CLI sandbox forbids binding sockets (any `server.listen()` leads to `EPERM`).
- **Workarounds**: We first mocked `fetch` with `node-mocks-http`, but matching streaming semantics, headers, and response bodies proved brittle.
- **Supertest migration**: Converting to supertest exercised the Express stack directly, but supertest still tries to bind sockets, triggering the sandbox blocks.
- **Opt-in execution**: The compromise is an environment flag (`MARKSERV_ENABLE_SUPERTEST=1`) to run the suites only where sockets are allowed; otherwise they auto-skip to keep default runs green.

## Role of the Sandbox
- **Port restrictions**: The sandbox guards the host system, disallowing `listen(0.0.0.0)` calls by default; any code path that opens a port fails unless we stub or skip it.
- **File-system isolation**: Some tests write fixtures; we make sure they clean up after themselves to avoid pollution.
- **Process spawning**: The sandbox allows child processes but may restrict network activity, so we prefer in-process adapters (e.g., `createMarkservApp`).

## Practical Takeaways
1. **Design for pure functions**: Share Express app creation logic (`createMarkservApp`) so tests can drive the app without starting the HTTP server.
2. **Plan for fallbacks**: Tests that depend on networking need an opt-in switch or a mock layer to operate in restricted environments.
3. **Document flags**: Add a note (`MARKSERV_ENABLE_SUPERTEST`) so contributors know how to enable the deeper coverage locally/CI while keeping sandbox runs stable.
4. **Avoid re-implementing HTTP**: Maintaining fetch shims that perfectly mimic streaming is harder than reusing existing test helpers (supertest, superagent) when possible.
5. **Guard cleanup**: When tests write to disk (e.g., temp files), ensure the cleanup happens even when assertions fail to keep sandbox state predictable.

## Recommended Workflow
- **Default (sandbox)**: Run `npm test` without extra flags; supertest suites skip, unit tests still cover core logic.
- **Extended (local/CI)**: Export `MARKSERV_ENABLE_SUPERTEST=1` and rerun tests to exercise full HTTP responses.
- **Future enhancements**: Consider building an HTTP shim compatible with supertest for environments that forbid sockets, or schedule a dedicated CI job with network access.
