FROM ultralytics/ultralytics:latest-python

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir uv
RUN uv pip install --system -e .

COPY . .

EXPOSE 8050
CMD ["sh", "-c", "gunicorn src.inference:server --bind 0.0.0.0:${PORT:-8050} --workers 1 --threads 4 --timeout 120"]