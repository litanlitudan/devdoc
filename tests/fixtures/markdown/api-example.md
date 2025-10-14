# API Route Examples

The `/api/` route provides direct file downloads suitable for command-line tools like `wget` and `curl`.

## Examples

### Using curl

```bash
# Download a file directly
curl http://localhost:8080/api/demo.js -o demo.js

# Download package.json
curl http://localhost:8080/api/package.json -o package.json

# Download a file from a subdirectory
curl http://localhost:8080/api/lib/server.js -o server.js
```

### Using wget

```bash
# Download a file directly
wget http://localhost:8080/api/demo.js

# Download package.json
wget http://localhost:8080/api/package.json

# Download with custom output name
wget http://localhost:8080/api/lib/server.js -O my-server.js
```

### Using fetch (Node.js)

```javascript
// Download a file
const response = await fetch('http://localhost:8080/api/package.json')
const content = await response.text()
console.log(content)
```

## How It Works

1. **Regular URL**: `http://localhost:8080/demo.js` → Renders file with syntax highlighting in browser
2. **API URL**: `http://localhost:8080/api/demo.js` → Downloads raw file directly

The API route:
- Sets `Content-Disposition: attachment` header
- Uses proper MIME types
- Works with all file types
- Returns 404 for non-existent files
- Returns 400 for directories
- Supports files with special characters in names
