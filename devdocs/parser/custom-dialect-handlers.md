# Adding Handlers For A Custom MLIR Dialect

Goal
- Improve visualization for a custom dialect when you have its MLIR sources: better names, region labeling, tensor tags, and attributes — while retaining generic fallback for unknown ops.

Integration Levels
- Generic (already works): Parser accepts unknown dialects via `allowUnregisteredDialects(true)` and renders nodes/edges/attrs generically.
- Dialect‑aware (recommended if sources available): Register the dialect and add small hooks to enhance names/regions/metadata.

1) Register Your Dialect (printing + parsing)
- File: `third_party/model-explorer/src/builtin-adapter/model_json_graph_convert.cc`
- Add your dialect to the registry used in `ConvertMlirToJson`:
```c++
mlir::DialectRegistry registry;
registry.insert<mlir::func::FuncDialect, /* ... existing ... */
               your_namespace::YourDialect>();
mlir::MLIRContext context(registry);
context.allowUnregisteredDialects(true);
```
- Build: ensure the dialect library is linked. In `third_party/model-explorer/src/builtin-adapter/BUILD`, add your cc_library to the `deps` of `model_json_graph_convert` and any target that needs your headers.

2) Add Dialect Detection + Naming
- File: `translate_helpers.cc`
- Add a helper similar to `IsStablehloDialect`:
```c++
inline bool IsYourDialect(mlir::Operation& op) {
  return llvm::isa<your_namespace::YourDialect>(op.getDialect());
}
```
- In `AddNodeInfo(...)`, add a branch to set label/name the way you want (e.g., derive hierarchical name from locations or attrs); otherwise it falls back to generic StableHLO‑style naming:
```c++
if (IsYourDialect(operation)) {
  // Example: preserve full op name as label and use NameLoc for hierarchy
  builder.SetNodeId(node_id_str);
  builder.SetNodeLabel(operation.getName().getStringRef());
  AddJaxNodeNameAndAttribute(operation, builder); // or custom logic
  return;
}
```

3) Handle Nested Regions (optional but valuable)
- Implement a region processor akin to `ProcessStablehloRegions`/`ProcessTosaRegions` and wire it in `MaybeAddNestedRegion(...)`:
```c++
absl::StatusOr<bool> ProcessYourDialectRegions(
  const std::function<absl::Status(absl::string_view, mlir::Region&)>& process_region,
  mlir::Operation& op) {
  return llvm::TypeSwitch<mlir::Operation*, absl::StatusOr<bool>>(&op)
    .Case<your_namespace::WhileOp>([&](your_namespace::WhileOp w) {
      RETURN_IF_ERROR(process_region("cond", w.getCond()));
      RETURN_IF_ERROR(process_region("body", w.getBody()));
      return true;
    })
    .Default([](mlir::Operation*) { return false; });
}
// In MaybeAddNestedRegion(...):
if (IsYourDialect(operation)) {
  ASSIGN_OR_RETURN(region_processed, ProcessYourDialectRegions(process_region, operation));
}
```
- Benefit: inner ops appear under meaningful namespaces instead of generic `(region_i)`.

4) Attribute Formatting / Tensor Tags (optional)
- Attributes: `AppendNodeAttrs(...)` already prints MLIR attributes. Add custom printers if you want to compact large lists or redact blobs.
- Tensor tags: If you have op schema (arg/result names), mirror the TFL pattern (`AddTensorTags(...)`): map op label → arg/result names and call `AppendAttrToMetadata(EdgeType::kInput/kOutput, idx, "__tensor_tag", name)`.

5) Subgraph Linking (if your ops call functions)
- For ops that reference symbols (e.g., function calls), append subgraph ids so users can jump:
```c++
if (auto flat_sym = llvm::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(attr_val)) {
  builder.AppendSubgraphId(flat_sym.getValue());
}
```

6) Build & Link Notes (Bazel)
- Add your dialect to the relevant `cc_library` deps in `builtin-adapter/BUILD` (e.g., `model_json_graph_convert`, `translate_helpers`). Ensure headers are included and the external repo providing the dialect is declared in `WORKSPACE` if needed.

Bazel examples
```python
# WORKSPACE
# Use http_archive/local_repository per your setup.
http_archive(
    name = "your_dialect_repo",
    urls = ["https://example.com/your-dialect/archive/main.zip"],
    strip_prefix = "your-dialect-main",
)

# third_party/your_dialect/BUILD
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "your_dialect",
    srcs = [
        "ir/your_dialect.cc",
        "ir/your_ops.cc",
    ],
    hdrs = [
        "ir/your_dialect.h",
        "ir/your_ops.h",
    ],
    includes = ["ir"],  # so #include "your_dialect.h" works
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    visibility = ["//visibility:public"],
)

# third_party/model-explorer/src/builtin-adapter/BUILD
cc_library(
    name = "model_json_graph_convert",
    srcs = ["model_json_graph_convert.cc"],
    hdrs = ["model_json_graph_convert.h"],
    deps = [
        # ... existing deps ...
        "//third_party/your_dialect:your_dialect",
    ],
)

cc_library(
    name = "translate_helpers",
    srcs = ["translate_helpers.cc"],
    hdrs = ["translate_helpers.h"],
    deps = [
        # ... existing deps ...
        "//third_party/your_dialect:your_dialect",
    ],
)
```

CMake examples (non-Bazel builds)
```cmake
# Set these to your local LLVM/MLIR install prefix or build directory exports
set(LLVM_DIR "/path/to/llvm/lib/cmake/llvm")
set(MLIR_DIR "/path/to/llvm/lib/cmake/mlir")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Your dialect library
add_library(YourDialect STATIC
  ir/your_dialect.cc
  ir/your_ops.cc
)
target_include_directories(YourDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/ir
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(YourDialect PUBLIC
  MLIRIR
  MLIRSupport
  # If your dialect exports a CMake target (e.g., MLIRYourDialect), link it too
  # MLIRYourDialect
)

# Example: extend a translator that depends on your dialect
add_library(ModelExplorerTranslateHelpers EXTENSION
  translate_helpers_extension.cc
)
target_link_libraries(ModelExplorerTranslateHelpers PUBLIC
  YourDialect
  MLIRIR
  MLIRSupport
)

# Tips:
# - Set CMAKE_PREFIX_PATH to include LLVM/MLIR install prefix, or set LLVM_DIR/MLIR_DIR.
# - Use LLVM_ENABLE_RTTI=ON when building LLVM/MLIR if your project needs RTTI.
# - For installed MLIR, link against MLIRTableGen and other components as needed.
```

Example extension stub (translate_helpers hooks)
```c++
// translate_helpers_extension.cc
// Minimal example showing dialect detection and region handling hooks.

#include "translate_helpers.h"              // Project helper APIs
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/TypeName.h"

// Replace with your dialect headers
#include "your_dialect/ir/your_dialect.h"
#include "your_dialect/ir/your_ops.h"

namespace tooling {
namespace visualization_client {

inline bool IsYourDialect(mlir::Operation& op) {
  return llvm::isa<your_namespace::YourDialect>(op.getDialect());
}

// Dialect-specific region processor; return true if handled.
absl::StatusOr<bool> ProcessYourDialectRegions(
    const std::function<absl::Status(absl::string_view, mlir::Region&)>& process_region,
    mlir::Operation& op) {
  return llvm::TypeSwitch<mlir::Operation*, absl::StatusOr<bool>>(&op)
      .Case<your_namespace::WhileOp>([&](your_namespace::WhileOp w) {
        RETURN_IF_ERROR(process_region("cond", w.getCond()));
        RETURN_IF_ERROR(process_region("body", w.getBody()));
        return true;
      })
      .Default([](mlir::Operation*) { return false; });
}

// Integration hints (patch existing functions in translate_helpers.cc):
//
// 1) In AddNodeInfo(...):
//    if (IsYourDialect(operation)) {
//      builder.SetNodeId(node_id_str);
//      builder.SetNodeLabel(operation.getName().getStringRef());
//      // Option A: reuse JAX-style hierarchical name from NameLoc
//      AddJaxNodeNameAndAttribute(operation, builder);
//      // Option B: write a custom naming function here
//      return;
//    }
//
// 2) In MaybeAddNestedRegion(...):
//    if (IsYourDialect(operation)) {
//      ASSIGN_OR_RETURN(region_processed,
//        ProcessYourDialectRegions(process_region, operation));
//    }

}  // namespace visualization_client
}  // namespace tooling
```

Custom naming from dialect attribute
```c++
// Example: derive hierarchical node name from a dialect-specific attribute.
// If not present, fall back to NameLoc (JAX-style) or leave empty.

#include "mlir/IR/BuiltinAttributes.h"

namespace tooling {
namespace visualization_client {

static void SetNameFromDialectAttrOrFallback(mlir::Operation& operation,
                                             GraphNodeBuilder& builder) {
  // Suppose ops in YourDialect may carry: your.namespace = "a/b/c"
  if (auto dict = operation.getAttrDictionary()) {
    if (auto nsAttr = dict.get("your.namespace").dyn_cast_or_null<mlir::StringAttr>()) {
      builder.SetNodeName(nsAttr.getValue());
      return;
    }
  }
  // Fallbacks: NameLoc (keeps parity with StableHLO/JAX path)
  AddJaxNodeNameAndAttribute(operation, builder);
}

// Usage inside AddNodeInfo(...)
// if (IsYourDialect(operation)) {
//   builder.SetNodeId(node_id_str);
//   builder.SetNodeLabel(operation.getName().getStringRef());
//   SetNameFromDialectAttrOrFallback(operation, builder);
//   return;
// }

}  // namespace visualization_client
}  // namespace tooling
```

7) Test & Verify
- Create small samples under `devdocs/parser/samples/` (with nested regions and attributes).
- Convert using the adapter wrapper:
```bash
python - << 'PY'
from ai_edge_model_explorer_adapter import _pywrap_convert_wrapper as me
cfg = me.VisualizeConfig(); cfg.const_element_count_limit = 128
print(me.ConvertMlirToJson(cfg, 'path/to/your.mlir')[:400])
PY
```
- Open in Model Explorer and confirm: meaningful labels, region names, input helper nodes, edges, and metadata display.

Fallback Guarantee
- If any handler is missing, the pipeline still parses thanks to `allowUnregisteredDialects(true)` and the generic region walker, so you can iterate safely.
