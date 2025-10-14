# Diff Rendering Example

Markserv now supports rendering diff and patch files with syntax highlighting!

## Example Diff Block

```diff
diff --git a/package.json b/package.json
index abc123..def456 100644
--- a/package.json
+++ b/package.json
@@ -10,7 +10,8 @@
   "scripts": {
     "test": "vitest",
-    "build": "node scripts/build.js"
+    "build": "node scripts/build.js",
+    "lint": "eslint . --fix"
   },
   "dependencies": {
```

## Features

- ✅ Renders `.diff` and `.patch` files with syntax highlighting
- ✅ Supports diff/patch code blocks in markdown files
- ✅ Color-coded additions (green) and deletions (red)
- ✅ Proper highlighting for file headers, hunk markers, and metadata
- ✅ Live reload support for diff files

## Supported Elements

- **File headers**: `---` and `+++` lines
- **Hunk markers**: `@@ -1,3 +1,4 @@`
- **Additions**: Lines starting with `+`
- **Deletions**: Lines starting with `-`
- **Diff command**: `diff --git` lines
- **Index lines**: `index abc123..def456`
- **File mode changes**: `new file mode`, `old mode`, etc.
- **Binary files**: `Binary files a/image.png and b/image.png differ`