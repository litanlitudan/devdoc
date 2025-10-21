# DevContainer Troubleshooting Guide

This document contains solutions to common devcontainer build and runtime issues.

## Build Failures

### User Creation Error (exit code: 4)

**Symptom:**

```
ERROR: failed to solve: process "/bin/sh -c groupadd --gid $USER_GID $USERNAME ..."
did not complete successfully: exit code: 4
```

**Cause:**
Exit code 4 from `groupadd`/`useradd` means the UID or GID already exists in the base Ubuntu image.

**Solution:**
The Dockerfile has been updated to handle this automatically. If you still encounter this error:

1. **Rebuild without cache:**

   ```bash
   # In VS Code: Press F1
   Dev Containers: Rebuild Container Without Cache
   ```

2. **Verify the fix is in place:**

   ```bash
   # Check that the Dockerfile uses fallback user creation
   grep -A 10 "Create non-root user" .devcontainer/Dockerfile
   ```

   You should see:

   ```dockerfile
   # Create group - try with the desired GID, fallback to auto-assigned GID if taken
   && (groupadd --gid $USER_GID $USERNAME 2>/dev/null || groupadd $USERNAME)
   # Create user - try with the desired UID, fallback to auto-assigned UID if taken
   && (useradd --uid $USER_UID --gid $USERNAME -m -s /bin/bash $USERNAME 2>/dev/null || \
       useradd --gid $USERNAME -m -s /bin/bash $USERNAME)
   ```

### Invalid User in chown Command

**Symptom:**

```
chown: invalid user: 'dev:dev'
ERROR: failed to solve: process "/bin/sh -c chown -R $USERNAME ..." exit code: 1
```

**Cause:**
ARG variables don't persist across multiple RUN commands in Dockerfile. The `$USERNAME` variable needs to be re-declared before use.

**Solution:**
The Dockerfile has been updated to re-declare the ARG before using it in chown:

```dockerfile
ARG USERNAME=dev
RUN git config --global --add safe.directory /workspace \
    && chown -R $USERNAME:$USERNAME /workspace
```

This ensures the variable is available in the build context where it's used.

3. **Alternative: Use a different UID/GID**

   If you need to use a different UID/GID, edit `.devcontainer/devcontainer.json`:

   ```json
   "build": {
     "dockerfile": "Dockerfile",
     "context": "..",
     "args": {
       "USER_UID": "1001",
       "USER_GID": "1001"
     }
   }
   ```

### LLVM/MLIR Not Found During Build

**Symptom:**

```
E: Unable to locate package llvm-18
E: Unable to locate package libmlir-18-dev
```

**Cause:**
The Ubuntu package repository might not have LLVM 18 available yet, or the repositories need to be updated.

**Solutions:**

1. **Update package lists:**
   Add `apt-get update` before the LLVM installation in Dockerfile.

2. **Use LLVM 17 instead:**

   ```bash
   # Edit .devcontainer/Dockerfile
   # Replace all instances of llvm-18 with llvm-17
   sed -i 's/llvm-18/llvm-17/g' .devcontainer/Dockerfile
   sed -i 's/mlir-18/mlir-17/g' .devcontainer/Dockerfile
   ```

3. **Install from LLVM APT repository:**
   Add the official LLVM repository before installing:
   ```dockerfile
   RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
       && add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-18 main" \
       && apt-get update
   ```

### Python pip Installation Failure (externally-managed-environment)

**Symptom:**

```
error: externally-managed-environment
ERROR: failed to solve: process "/bin/sh -c pip3 install ..." exit code: 1
```

**Cause:**
Ubuntu 24.04 implements PEP 668, which prevents pip from installing packages system-wide to avoid conflicts with system-managed packages.

**Solution:**
The Dockerfile has been updated to use `--break-system-packages` flag:

```dockerfile
RUN pip3 install --no-cache-dir --break-system-packages \
    onnx>=1.12.0 \
    numpy
```

This is safe in Docker containers since we control the entire environment.

**Alternative Solutions:**

1. **Use virtual environment (not recommended for containers):**

   ```dockerfile
   RUN python3 -m venv /opt/venv
   ENV PATH="/opt/venv/bin:$PATH"
   RUN pip3 install onnx>=1.12.0 numpy
   ```

2. **Install via apt (if packages are available):**
   ```dockerfile
   RUN apt-get install -y python3-numpy python3-onnx
   ```

### Node.js Installation Failure

**Symptom:**

```
E: Unable to locate package nodejs
```

**Cause:**
The NodeSource repository might not have been added correctly.

**Solution:**
Verify the GPG key and repository setup:

```dockerfile
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | \
       gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] \
       https://deb.nodesource.com/node_20.x nodistro main" | \
       tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update
```

### Docker Out of Disk Space

**Symptom:**

```
no space left on device
```

**Solution:**

```bash
# Check Docker disk usage
docker system df

# Clean up unused containers, images, and volumes
docker system prune -a --volumes

# Or manually remove specific items
docker container prune
docker image prune -a
docker volume prune
```

## Runtime Issues

### Permission Denied Errors

**Symptom:**

```
Permission denied when accessing /workspace or npm install fails
```

**Solutions:**

1. **Check user ownership:**

   ```bash
   # Inside container
   ls -la /workspace
   # Should show dev:dev as owner
   ```

2. **Fix ownership if needed:**

   ```bash
   # Run as root (sudo -i)
   chown -R dev:dev /workspace
   ```

3. **Rebuild container:**
   The Dockerfile automatically sets correct ownership, so rebuilding should fix it.

### npm install Fails

**Symptom:**

```
npm ERR! code EACCES
npm ERR! syscall access
npm ERR! path /workspace/node_modules
```

**Solutions:**

1. **Clear npm cache:**

   ```bash
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Check Node.js version:**

   ```bash
   node --version  # Should be 20.x
   npm --version
   ```

3. **Reinstall Node.js:**
   ```bash
   sudo apt-get update
   sudo apt-get install --reinstall nodejs
   ```

### LLVM/MLIR Tools Not in PATH

**Symptom:**

```bash
llvm-config: command not found
```

**Solutions:**

1. **Check environment variables:**

   ```bash
   echo $LLVM_DIR        # Should be /usr/lib/llvm-18
   echo $PATH            # Should include /usr/lib/llvm-18/bin
   ```

2. **Source environment in shell:**

   ```bash
   export LLVM_DIR=/usr/lib/llvm-18
   export PATH="${LLVM_DIR}/bin:${PATH}"
   ```

3. **Verify LLVM installation:**
   ```bash
   ls -la /usr/lib/llvm-18/bin/
   dpkg -l | grep llvm
   ```

### Port Already in Use

**Symptom:**

```
Error: listen EADDRINUSE: address already in use :::8642
```

**Solutions:**

1. **Check what's using the port:**

   ```bash
   lsof -ti :8642
   # Or on the host:
   lsof -ti :8642 | xargs kill
   ```

2. **Use different ports:**
   Edit `.devcontainer/devcontainer.json`:
   ```json
   "forwardPorts": [8643, 3685, 35730]
   ```

### MLIR Parser Build Fails

**Symptom:**

```
CMake Error: Could not find MLIR
```

**Solutions:**

1. **Verify LLVM installation:**

   ```bash
   llvm-config --version
   llvm-config --cmakedir
   ```

2. **Set CMAKE_PREFIX_PATH explicitly:**

   ```bash
   cd src/mlir/build
   cmake -DCMAKE_PREFIX_PATH=/usr/lib/llvm-18 ..
   ninja
   ```

3. **Check for nlohmann-json:**
   ```bash
   dpkg -l | grep nlohmann-json3-dev
   # If missing:
   sudo apt-get install nlohmann-json3-dev
   ```

## Container Won't Start

### VS Code Can't Connect

**Symptoms:**

- Container builds successfully but VS Code can't connect
- "Failed to connect to remote extension host server" error

**Solutions:**

1. **Restart Docker:**

   ```bash
   # On macOS/Windows
   Restart Docker Desktop

   # On Linux
   sudo systemctl restart docker
   ```

2. **Remove old containers:**

   ```bash
   docker container ls -a | grep devdoc
   docker container rm <container-id>
   ```

3. **Rebuild container:**
   ```
   Dev Containers: Rebuild Container
   ```

### Extensions Not Installing

**Symptom:**
Extensions listed in devcontainer.json don't install.

**Solutions:**

1. **Manual installation:**

   ```bash
   # Inside container
   code --install-extension ms-vscode.cpptools
   code --install-extension ms-vscode.cmake-tools
   ```

2. **Check extension IDs:**
   Verify the extension IDs in `.devcontainer/devcontainer.json` are correct.

3. **Internet connectivity:**
   ```bash
   # Inside container
   curl -I https://marketplace.visualstudio.com
   ```

## Getting Help

If you encounter an issue not covered here:

1. **Check the logs:**
   - Build logs: Look for ERROR or FAILED messages
   - Runtime logs: Check Docker container logs

2. **Search for similar issues:**
   - [Dev Containers Issues](https://github.com/microsoft/vscode-dev-containers/issues)
   - [Docker Issues](https://github.com/docker/for-mac/issues)

3. **Collect diagnostic info:**

   ```bash
   # Docker version
   docker --version
   docker-compose --version

   # VS Code version
   code --version

   # Container status
   docker ps -a

   # Container logs
   docker logs <container-id>
   ```

4. **File an issue:**
   Include the above diagnostic information when reporting problems.

## Useful Commands

```bash
# View container logs
docker logs <container-name>

# Enter container as root
docker exec -it -u root <container-name> /bin/bash

# Inspect container
docker inspect <container-name>

# View Docker disk usage
docker system df

# Clean up everything (WARNING: removes all Docker data)
docker system prune -a --volumes

# Rebuild devcontainer without cache
# In VS Code: F1 → "Dev Containers: Rebuild Container Without Cache"
```

## Prevention Tips

1. **Regular cleanup:**

   ```bash
   # Weekly maintenance
   docker system prune -a
   ```

2. **Monitor disk space:**

   ```bash
   docker system df
   ```

3. **Use .dockerignore:**
   Ensure `.devcontainer/.dockerignore` is configured to exclude unnecessary files.

4. **Keep Docker updated:**
   Update Docker Desktop regularly for bug fixes and improvements.

5. **Use specific versions:**
   Pin specific versions in Dockerfile to ensure reproducible builds.
