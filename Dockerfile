# Research Intelligence: the dashboard server and the research0 command line, for the EC2 deployment (see deploy/).
# The package is installed in editable mode on purpose: migrations/ and config/ are found next to src/ (as in the repository).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY migrations ./migrations
COPY config ./config
RUN pip install -e . && useradd --system --uid 10001 --home-dir /app research && mkdir -p /app/reports && chown research /app/reports

USER research
EXPOSE 8765
ENTRYPOINT ["research0"]
CMD ["--help"]
