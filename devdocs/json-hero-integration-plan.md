# JSON Hero Integration Plan

## Overview

This document outlines the plan to integrate JSON Hero UI into markserv, following the existing model-explorer integration pattern. When a JSON file is accessed via `/json/<path>`, markserv will render the JSON Hero interface to provide an enhanced viewing experience.

## Current State Analysis

### Model Explorer Integration Pattern

The existing MLIR/ONNX integration provides a proven pattern we'll follow:

**Request Flow** (server.ts:1327-1445):
1. Special route handler: `/model-explorer/<file-path>`
2. File validation: Check extension (`.mlir`, `.onnx`) and existence
3. File reading: Load file content as buffer
4. Conversion: Transform format to Model Explorer graph JSON
5. Template rendering: Inject graph data into `model-explorer.html` template
6. Static resources: Serve Model Explorer scripts from `/lib/model-explorer/`

**Key Components**:
- **Route Handler**: Express middleware in `server.ts` for `/model-explorer/` path
- **Conversion Module**: `mlir-to-graph.ts` / `onnx-to-graph.ts` for data transformation
- **HTML Template**: `lib/templates/model-explorer.html` with embedded JavaScript loader
- **Static Files**: Browser-compatible scripts in `lib/model-explorer/` directory
- **File Type Support**: Added to `fileTypes.mlir` and `fileTypes.onnx` arrays
- **LiveReload**: MLIR/ONNX files added to watch list for auto-refresh

### JSON Hero Structure

JSON Hero is a Remix-based web application with the following characteristics:

**Technology Stack**:
- **Framework**: Remix (React-based full-stack framework)
- **Runtime**: Cloudflare Workers (can run in Node.js with miniflare)
- **UI**: React + TailwindCSS
- **Build**: esbuild for bundling, tailwindcss for styling

**Key Features**:
- Column View, Tree View, Editor View for JSON navigation
- Automatic content type inference (dates, URLs, colors, etc.)
- Search functionality with fuzzy matching
- JSON Schema inference
- Related values scanning
- Theme support (light/dark)

**Build Process** (package.json):
```bash
npm run build:css    # Compile Tailwind CSS
npm run build:search # Bundle search worker
remix build          # Build Remix app
```

**Entry Points**:
- `app/root.tsx`: Root component with providers and theme setup
- `app/entry.client.tsx`: Client-side hydration
- `app/entry.server.tsx`: Server-side rendering
- `app/routes/`: Route handlers for different views

## Integration Architecture

### Approach: Embedded Static Build

Similar to model-explorer, we'll use JSON Hero as an embedded static application:

1. **Pre-build JSON Hero**: Build JSON Hero into static assets during setup
2. **Embed in markserv**: Copy built assets to `lib/jsonhero/` directory
3. **Serve via Express**: Add route handler for `/json/<path>` URLs
4. **Data Injection**: Pass JSON data via inline script tag or API endpoint

### Directory Structure

```
markserv/
├── lib/
│   ├── jsonhero/              # JSON Hero static build output
│   │   ├── index.html         # Modified entry point
│   │   ├── assets/            # Compiled JS, CSS, images
│   │   │   ├── entry.*.js     # Client-side bundle
│   │   │   ├── root.*.js      # Root component bundle
│   │   │   └── *.css          # Compiled stylesheets
│   │   └── worker.js          # Search worker (optional)
│   ├── templates/
│   │   └── json-hero.html     # Wrapper template
│   └── server.ts              # Add JSON Hero route handler
├── third_party/
│   └── jsonhero-web/          # Source for rebuilding (existing)
└── scripts/
    └── build-jsonhero.sh      # Build script to prepare assets
```

### Implementation Components

#### 1. Build Script (`scripts/build-jsonhero.sh`)

```bash
#!/bin/bash
# Build JSON Hero and copy to lib directory

set -e

cd third_party/jsonhero-web

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    npm install
fi

# Build JSON Hero
npm run build:css
npm run build:search
npm run build

# Create output directory
mkdir -p ../../lib/jsonhero/assets

# Copy built assets
cp -r public/build/* ../../lib/jsonhero/assets/
cp public/entry.worker.js ../../lib/jsonhero/
cp app/tailwind.css ../../lib/jsonhero/assets/

echo "JSON Hero build complete"
```

#### 2. HTML Template (`lib/templates/json-hero.html`)

Similar to `model-explorer.html`, create a wrapper template:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JSON Hero - {{filename}}</title>
    <link rel="stylesheet" href="/lib/jsonhero/assets/tailwind.css">
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        }
        .header {
            background: #1f2937;
            color: white;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #374151;
        }
        .header h1 {
            font-size: 18px;
            margin: 0;
        }
        .btn {
            padding: 6px 12px;
            background: #374151;
            border: 1px solid #4b5563;
            border-radius: 4px;
            color: white;
            text-decoration: none;
            font-size: 13px;
        }
        .btn:hover {
            background: #4b5563;
        }
        #json-hero-root {
            height: calc(100vh - 48px);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>JSON Hero - {{filename}}</h1>
        <div>
            <a href="{{backUrl}}" class="btn">← Back</a>
            <a href="#" onclick="window.location.reload()" class="btn">Refresh</a>
        </div>
    </div>

    <div id="json-hero-root"></div>

    <!-- JSON Data -->
    <script id="json-data" type="application/json">{{jsonData}}</script>

    <!-- JSON Hero Application -->
    <script src="/lib/jsonhero/assets/entry.*.js"></script>
    <script src="/lib/jsonhero/assets/root.*.js"></script>

    <script>
        // Initialize JSON Hero with embedded data
        const jsonDataElement = document.getElementById('json-data');
        const jsonData = JSON.parse(jsonDataElement.textContent);

        // Create JSON document for JSON Hero
        window.__REMIX_DATA__ = {
            jsonDocument: {
                title: '{{filename}}',
                content: jsonData,
                readOnly: true
            }
        };

        // JSON Hero will hydrate from this data
    </script>
</body>
</html>
```

#### 3. Route Handler in `server.ts`

Add handler similar to model-explorer (after line ~1445):

```typescript
// Handle JSON Hero route (/json)
if (decodedUrl.startsWith('/json/') || decodedUrl === '/json') {
    const jsonFilePath = decodedUrl === '/json' ? '' : decodedUrl.substring(6) // Remove '/json/'

    if (!jsonFilePath || jsonFilePath === '') {
        res.status(400).send('Please specify a JSON file path, e.g., /json/data.json')
        return
    }

    // Resolve the actual file path
    const actualFilePath = path.normalize(path.join(dir, jsonFilePath))

    // Check if file exists and has .json extension
    const isJSON = actualFilePath.endsWith('.json')

    if (!isJSON) {
        res.status(400).send('Only JSON (.json) files are supported')
        return
    }

    fs.stat(actualFilePath, (err, stats) => {
        if (err || !stats.isFile()) {
            res.status(404).send('JSON file not found')
            return
        }

        // Read JSON file
        fs.readFile(actualFilePath, 'utf8', (readErr, jsonContent) => {
            if (readErr) {
                console.error('Error reading JSON file:', readErr)
                res.status(500).send('Failed to read JSON file')
                return
            }

            // Validate JSON
            try {
                JSON.parse(jsonContent)
            } catch (parseErr) {
                res.status(400).send('Invalid JSON file')
                return
            }

            // Load and render the JSON Hero template
            const templatePath = path.join(libPath, 'templates/json-hero.html')
            fs.readFile(templatePath, 'utf8', (templateErr, template) => {
                if (templateErr) {
                    console.error('Error loading JSON Hero template:', templateErr)
                    res.status(500).send('Failed to load JSON Hero template')
                    return
                }

                // Prepare template data
                const backUrl = path.dirname(decodedUrl) || '/'
                const templateData = {
                    filename: path.basename(jsonFilePath),
                    jsonData: jsonContent, // Already stringified JSON
                    backUrl
                }

                // Compile and render template
                const compiledTemplate = handlebars.compile(template)
                const html = compiledTemplate(templateData)

                res.setHeader('Content-Type', 'text/html')
                res.send(html)
            })
        })
    })
    return
}
```

#### 4. Static File Serving

Add route to serve JSON Hero assets (similar to model-explorer at line ~1508):

```typescript
// Serve JSON Hero static files
app.use('/lib/jsonhero', express.static(path.join(libPath, 'jsonhero')))
```

#### 5. File Type Configuration

Add JSON to watched file types (around line ~740):

```typescript
watchExtensions: [
    '.md',
    '.markdown',
    '.mdown',
    '.txt',
    '.css',
    '.less',
    '.js',
    '.json',  // Add JSON to watch list
    '.gif',
    '.png',
    '.jpg',
    '.jpeg',
    '.mlir',
    '.onnx'
]
```

## Implementation Phases

### Phase 1: Build Infrastructure (Setup)
- [ ] Create `scripts/build-jsonhero.sh` build script
- [ ] Test JSON Hero build process
- [ ] Add build script to npm scripts in package.json
- [ ] Document build requirements in CLAUDE.md

### Phase 2: Template & Static Assets (Core Integration)
- [ ] Create `lib/templates/json-hero.html` template
- [ ] Build JSON Hero and copy assets to `lib/jsonhero/`
- [ ] Test asset serving with simple HTML page

### Phase 3: Server Integration (Route Handler)
- [ ] Add `/json/` route handler in server.ts
- [ ] Implement JSON file validation and reading
- [ ] Integrate template rendering with handlebars
- [ ] Add static file serving for `/lib/jsonhero/`

### Phase 4: Testing & Refinement
- [ ] Test with various JSON files (small, large, nested)
- [ ] Verify LiveReload works with JSON files
- [ ] Test error handling (invalid JSON, missing files)
- [ ] Verify back navigation and refresh functionality

### Phase 5: Documentation
- [ ] Update CLAUDE.md with JSON Hero integration details
- [ ] Add usage examples to README.md
- [ ] Document troubleshooting steps
- [ ] Create test files in tests/ directory

## Alternative Approach: API-Based Loading

If static embedding proves complex due to Remix's architecture, consider:

**Hybrid Approach**:
1. Serve JSON Hero as standalone app on a sub-path (e.g., `/lib/jsonhero/`)
2. Create API endpoint `/api/json/<path>` that serves JSON data
3. JSON Hero loads via iframe or redirect with query parameter: `/lib/jsonhero/?url=/api/json/data.json`

**Pros**:
- Simpler integration (less template modification)
- Preserves full JSON Hero functionality
- Easier to update JSON Hero version

**Cons**:
- Requires running JSON Hero dev server alongside markserv
- More complex navigation flow
- Additional HTTP requests for data loading

## Technical Considerations

### JSON Hero Remix Specifics

**Challenge**: JSON Hero uses Remix which expects:
- Server-side rendering with loaders
- Client-side hydration
- Router integration
- Session management

**Solution**:
- Use Remix's static export capability or build output
- Modify `entry.client.tsx` to accept data from window object
- Strip out routing, keep just the viewer components
- Simplify to single-page app with data injection

### Data Size Limits

**Concern**: Large JSON files may cause:
- Memory issues when loading entire file
- Slow template rendering
- Browser performance degradation

**Mitigation**:
- Add file size check (warn if >10MB)
- Stream large files instead of loading entirely
- Consider lazy loading for huge arrays/objects
- Provide download option for oversized files

### JSON Hero Customization

**Required Changes** to JSON Hero source:
- Disable create/upload functionality (read-only mode)
- Remove server dependencies (KV storage, auth)
- Strip out routing logic
- Simplify entry points for static embedding
- Remove analytics/tracking code

### Browser Compatibility

- Ensure JSON Hero works in modern browsers (Chrome, Firefox, Safari, Edge)
- Test with JavaScript modules and ES6+ features
- Verify CSS compatibility (Grid, Flexbox, Custom Properties)

## Success Criteria

- [ ] JSON files accessible via `/json/<path>` route
- [ ] JSON Hero UI loads and displays JSON correctly
- [ ] All view modes functional (Column, Tree, Editor)
- [ ] Search functionality works
- [ ] LiveReload triggers on JSON file changes
- [ ] Error handling for invalid JSON files
- [ ] Performance acceptable for JSON files up to 5MB
- [ ] Documentation complete and accurate

## Open Questions

1. **JSON Hero Simplification**: How much of JSON Hero do we need to strip out? Can we use it as-is or need significant modifications?

2. **Build Complexity**: Should we commit built JSON Hero assets to git, or rebuild on `npm install`?

3. **Update Strategy**: How do we handle JSON Hero updates? Manual rebuild or automated process?

4. **Performance**: What's the upper limit for JSON file size before UX degrades significantly?

5. **Feature Scope**: Which JSON Hero features are essential vs. nice-to-have?

## References

- Model Explorer Integration: `lib/server.ts:1327-1445`
- Model Explorer Template: `lib/templates/model-explorer.html`
- JSON Hero Source: `third_party/jsonhero-web/`
- JSON Hero README: `third_party/jsonhero-web/README.md`
- JSON Hero Development Guide: `third_party/jsonhero-web/DEVELOPMENT.md`

## Timeline Estimate

- **Phase 1** (Build Infrastructure): 2-3 hours
- **Phase 2** (Template & Assets): 3-4 hours
- **Phase 3** (Server Integration): 4-5 hours
- **Phase 4** (Testing & Refinement): 3-4 hours
- **Phase 5** (Documentation): 2-3 hours

**Total Estimate**: 14-19 hours

## Risk Assessment

**High Risk**:
- JSON Hero Remix architecture may not work in static embedding
- Build complexity could be higher than expected

**Medium Risk**:
- Performance issues with large JSON files
- Browser compatibility challenges

**Low Risk**:
- Route handler integration (proven pattern from model-explorer)
- Template rendering (standard handlebars approach)
