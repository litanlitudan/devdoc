# Use Ubuntu 24.04 LTS as base image
FROM ubuntu:24.04

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    # Build essentials for C++ compilation
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    nlohmann-json3-dev \
    # Python for optional ONNX support
    python3 \
    python3-pip \
    python3-venv \
    # LLVM/Clang/MLIR for native parser development
    llvm-18 \
    llvm-18-dev \
    llvm-18-tools \
    libmlir-18-dev \
    mlir-18-tools \
    clang-18 \
    lld-18 \
    # Version control and utilities
    git \
    curl \
    wget \
    vim \
    less \
    ca-certificates \
    gnupg \
    # JSON processing
    jq \
    # Cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS from NodeSource
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create user and group, handling cases where UID/GID might already exist
RUN apt-get update \
    && apt-get install -y sudo \
    # Create group - try with the desired GID, fallback to auto-assigned GID if taken
    && (groupadd --gid $USER_GID $USERNAME 2>/dev/null || groupadd $USERNAME) \
    # Create user - try with the desired UID, fallback to auto-assigned UID if taken
    && (useradd --uid $USER_UID --gid $USERNAME -m -s /bin/bash $USERNAME 2>/dev/null || \
        useradd --gid $USERNAME -m -s /bin/bash $USERNAME) \
    # Add sudo permissions
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # Cleanup
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install global npm packages
RUN npm install -g \
    typescript \
    ts-node \
    nodemon \
    npm-check-updates

# Install Python dependencies for optional ONNX support
# Use --break-system-packages to bypass PEP 668 externally-managed-environment protection
# This is safe in Docker containers where we control the entire environment
RUN pip3 install --no-cache-dir --break-system-packages \
    onnx>=1.12.0 \
    numpy

# Configure LLVM/MLIR environment
ENV LLVM_DIR=/usr/lib/llvm-18
ENV PATH="${LLVM_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${LLVM_DIR}/lib:${LD_LIBRARY_PATH}"
ENV CMAKE_PREFIX_PATH="${LLVM_DIR}:${CMAKE_PREFIX_PATH}"

# Set up alternatives for LLVM tools
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 \
    && update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-18 100

# Set up workspace directory
WORKDIR /workspace

# Configure git and set workspace ownership
# Do this in a single RUN to have access to USERNAME variable
ARG USERNAME=dev
RUN git config --global --add safe.directory /workspace \
    && chown -R $USERNAME:$USERNAME /workspace

# Switch to non-root user
USER $USERNAME

# Set up shell environment for dev user
RUN echo 'alias ll="ls -lah"' >> ~/.bashrc \
    && echo 'alias gs="git status"' >> ~/.bashrc \
    && echo 'alias gd="git diff"' >> ~/.bashrc \
    && echo 'export PS1="\[\033[01;32m\]\u@devcontainer\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> ~/.bashrc

# Default command
CMD ["/bin/bash"]
