# Stage 1: Node.js for frontend build + concurrently
FROM node:20-slim AS node-base

# Stage 2: Python + Node combined
FROM python:3.11-slim

# Copy Node.js from node image
COPY --from=node-base /usr/local/bin/node /usr/local/bin/node
COPY --from=node-base /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
  && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# Install build tools for Python C extensions (psutil etc) + procps for concurrently
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev procps && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package-lock.json ./frontend/
COPY backend/pyproject.toml backend/uv.lock ./backend/

# Install deps
RUN npm ci \
  && npm ci --prefix frontend \
  && cd backend && uv sync --frozen

# Copy source
COPY . .

EXPOSE 3000 5001

CMD ["npm", "run", "dev"]
