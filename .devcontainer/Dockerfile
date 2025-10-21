# Use Ubuntu 24.04 LTS as base image
FROM ubuntu:24.04

# Build arguments
ARG TZ=America/Los_Angeles
ARG CLAUDE_CODE_VERSION=latest
ARG GIT_DELTA_VERSION=0.18.2
ARG ZSH_IN_DOCKER_VERSION=1.2.0

# Set timezone and non-interactive frontend for apt
ENV TZ="$TZ"
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
    nano \
    less \
    ca-certificates \
    gnupg \
    # JSON processing
    jq \
    # Claude Code development tools
    procps \
    sudo \
    fzf \
    zsh \
    man-db \
    unzip \
    gnupg2 \
    gh \
    iptables \
    ipset \
    iproute2 \
    dnsutils \
    aggregate \
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

# Install git-delta for better git diffs
RUN ARCH=$(dpkg --print-architecture) && \
    wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# Create non-root user
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create user and group, handling cases where UID/GID might already exist
# Create group - try with the desired GID, fallback to auto-assigned GID if taken
# Create user - try with the desired UID, fallback to auto-assigned UID if taken
# Add sudo permissions
RUN (groupadd --gid $USER_GID $USERNAME 2>/dev/null || groupadd $USERNAME) \
    && (useradd --uid $USER_UID --gid $USERNAME -m -s /bin/bash $USERNAME 2>/dev/null || \
        useradd --gid $USERNAME -m -s /bin/bash $USERNAME) \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Ensure npm global directory is accessible
RUN mkdir -p /usr/local/share/npm-global && \
    chown -R $USERNAME:$USERNAME /usr/local/share

# Persist bash history
RUN mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory

# Set DEVCONTAINER environment variable
ENV DEVCONTAINER=true

# Create workspace and config directories
RUN mkdir -p /workspace /home/$USERNAME/.claude && \
    chown -R $USERNAME:$USERNAME /workspace /home/$USERNAME/.claude

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
ARG USERNAME=dev
RUN git config --global --add safe.directory /workspace \
    && chown -R $USERNAME:$USERNAME /workspace

# Switch to non-root user
USER $USERNAME

# Set up npm global directory for user
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

# Set default shell and editors
ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

# Set up fzf key bindings and completion for zsh
RUN mkdir -p ~/.fzf && \
    git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && \
    ~/.fzf/install --all --no-update-rc

# Install zsh-in-docker with powerline10k theme and plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
    -p git \
    -a "[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh" \
    -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    -a "alias ll='ls -lah'" \
    -a "alias gs='git status'" \
    -a "alias gd='git diff'" \
    -x

# Install global npm packages
RUN npm install -g \
    typescript \
    ts-node \
    nodemon \
    npm-check-updates \
    @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
    openai-cli

# Default command
CMD ["/bin/zsh"]
