FROM python:3.10-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

# Copy application code
COPY shellhacks/ ./shellhacks/

# Expose MCP server port
EXPOSE 8000

# Run MCP server
CMD ["python", "-m", "shellhacks.mcp_server"]