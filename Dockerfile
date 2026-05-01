FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY tests /workspace/tests
COPY scripts /workspace/scripts

RUN pip install --no-cache-dir -e ".[dev,experiments]"

CMD ["python", "-m", "pytest", "tests", "-v"]
