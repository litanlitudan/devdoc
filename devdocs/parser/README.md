# Parser Implementation Reports

This folder documents how Model Explorer integrations handle parsing and graph conversion in this repository.

- MLIR parsing and graph conversion: see `mlir.md`
- ONNX parsing and graph conversion: see `onnx.md`
- PyTorch ExportedProgram parsing: see `pytorch.md`
- Custom dialect sample & verification: see `custom-dialect.md` (sample under `samples/`).
- Custom dialect handlers (with sources): see `custom-dialect-handlers.md` for adding registry/region/name hooks.
- Highlighting grammar overview: see `highlightjs-mlir.md` for how Highlight.js tokenizes MLIR in docs.
- Universal parser design: see `universal-mlir-parser-design.md` for a tolerant, extensible graphization pipeline.

Each report highlights entry points, pipeline stages, graph-building logic, supported dialects/types, and noteworthy behaviors for debugging and extension. Mermaid flowcharts are included for quick visualization.
