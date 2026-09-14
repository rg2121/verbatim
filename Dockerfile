# A reproducible environment for the tool, not a production deployment: this
# is a local-first assistant whose corpus lives on one machine, and the image
# exists so a reader can run `search`, `ask` or `report` against the sample
# corpus without installing Python, uv, or ONNX runtime first.
FROM python:3.12-slim

# curl fetches uv. No compiler: every dependency - PyMuPDF, ONNX runtime,
# sqlite-vec - ships a manylinux wheel. (An earlier version installed a
# toolchain and purged it before `uv sync` ever ran, so it was never usable.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
# Dependencies before source, so editing a module does not re-resolve them.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY evals ./evals
COPY sample ./sample

# The corpus is mounted, not baked: it is 26 GB of other people's papers.
ENV VERBATIM_STATE_PATH=/data/corpus.db \
    VERBATIM_PDF_DIR=/data/pdfs \
    VERBATIM_HTML_DIR=/data/html \
    VERBATIM_LOG_FORMAT=json \
    OLLAMA_HOST=http://ollama:11434
VOLUME ["/data"]

# --no-dev matches the sync above. Without it `uv run` treats the dev group
# as missing and installs pytest on the container's first command, which
# needs a network the container may not have.
ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "verbatim"]
CMD ["stats"]
