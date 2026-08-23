FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home appuser

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=10001:10001 app ./app
COPY --chown=10001:10001 web ./web

RUN mkdir -p /app/data \
    && chown -R 10001:10001 /app/data

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).close()"]

CMD ["python", "-m", "app.server"]
