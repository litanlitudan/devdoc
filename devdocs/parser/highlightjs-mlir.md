# highlightjs-mlir: Syntax Highlighting Approach

Location
- `third_party/highlightjs-mlir/src/mlir.js` (ES module), with built assets under `dist/`.
- Loaded by Highlight.js to colorize MLIR in Markdown/HTML.

Goal
- Provide readable token coloring for common MLIR constructs; not a full parser.

Tokenization Strategy
- Keywords: `func`, `module`, `br`, `cond_br`, `return`.
- Primitive types: `iN`, `f16/32/64`, `bf16` via `PRIMITIVE_TYPES` mode.
- Dialect/types: `!<id>` as `type` (e.g., `!tf.index`).
- Shaped types: `(memref|tensor|vector)<...>` recognizing:
  - Dimensions: `*x` or repeated `(<num|?> x)+` as `number`.
  - Nested types via `self` recursion and `PRIMITIVE_TYPES`.
  - Layout spec variants include semi-affine maps.
- Affine maps/Layouts: `affine_map< (d..)->(d..) >` and raw `(…) -> (…)` captured as `attr/type`.
- Symbols:
  - SSA values: `%id`, `%id:#` with result indices.
  - Block labels: `^label`.
  - Attributes/IDs: `#id`.
- Titles: function/global identifiers `@name` or `@number`.
- Literals & comments: standard HLJS `C_NUMBER_MODE`, `QUOTE_STRING_MODE`, `C_LINE_COMMENT_MODE`.

Notable Design Choices
- Uses simple regex modes; no semantic validation or dialect registration.
- `self` recursion inside shaped types enables nested forms like `tensor<4xvector<10xf32>>`.
- Semi-affine map highlighting is shallow; complex affine expressions aren’t parsed.

Limitations
- No context-sensitive parsing (regions, ops, operands vs. results, attributes by type).
- Affine/polyhedral syntax and complex attributes are only partially recognized.
- Location annotations (e.g., `loc(...)`) are not specially tokenized.

Example (simplified)
```
%t2 = "std.dim"(%t){index = 2} : (tensor<4x4x?xvector<10xf32>>) -> !tf.index
```
- `%t2`, `%t` → symbol; `"std.dim"` → string; `index = 2` → attr/number; `tensor<...>` and `!tf.index` → type.

Usage
- Include the built script `dist/mlir.min.js` after Highlight.js; it auto-registers and enables `hljs.highlightAll()`.
