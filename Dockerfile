FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY defense_assistant ./defense_assistant
RUN pip install --no-cache-dir ".[server]"

# 지식 베이스·사용자 DB·감사 로그는 볼륨으로 마운트한다
COPY data ./data
RUN mkdir -p storage audit
VOLUME ["/app/data/docs", "/app/storage", "/app/audit"]

EXPOSE 8000
CMD ["defense-ai", "serve", "--host", "0.0.0.0", "--port", "8000"]
