# Syntax Highlighting Test

Testing syntax highlighting for Dockerfile, Diff, and Dart languages.

## Dockerfile

```dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application files
COPY . .

# Expose port
EXPOSE 8642

# Set environment variables
ENV NODE_ENV=production
ENV PORT=8642

# Start the application
CMD ["npm", "start"]
```

## Diff

```diff
diff --git a/lib/server.ts b/lib/server.ts
index abc123..def456 100644
--- a/lib/server.ts
+++ b/lib/server.ts
@@ -30,7 +30,10 @@
 import hljs from 'highlight.js/lib/core'
 import hljsMLIR from 'highlightjs-mlir'
+import hljsDockerfile from 'highlight.js/lib/languages/dockerfile'
+import hljsDiff from 'highlight.js/lib/languages/diff'
+import hljsDart from 'highlight.js/lib/languages/dart'

-// Register MLIR language for highlight.js
+// Register languages for highlight.js
 hljs.registerLanguage('mlir', hljsMLIR)
+hljs.registerLanguage('dockerfile', hljsDockerfile)
+hljs.registerLanguage('diff', hljsDiff)
+hljs.registerLanguage('dart', hljsDart)
```

## Dart

```dart
import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              'You have pushed the button this many times:',
            ),
            Text(
              '$_counter',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        tooltip: 'Increment',
        child: const Icon(Icons.add),
      ),
    );
  }
}
```

## Combined Example

Here's a Dockerfile that builds a Dart application:

```dockerfile
# Use Dart official image
FROM dart:stable AS build

# Set up working directory
WORKDIR /app

# Copy pubspec files
COPY pubspec.* ./

# Get dependencies
RUN dart pub get

# Copy source code
COPY . .

# Compile to executable
RUN dart compile exe bin/main.dart -o bin/server

# Build minimal runtime image
FROM scratch
COPY --from=build /runtime/ /
COPY --from=build /app/bin/server /app/bin/

# Start server
EXPOSE 8080
CMD ["/app/bin/server"]
```

This demonstrates that markserv now supports syntax highlighting for:
- ✅ Dockerfile
- ✅ Diff
- ✅ Dart
