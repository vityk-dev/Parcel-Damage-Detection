FROM ultralytics/ultralytics:latest-python

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv


COPY pyproject.toml ./


# Copy source code before editable install
COPY . .

# Install your package + deps into system (Render/Docker friendly)
RUN uv pip install --system -e .

EXPOSE 7860

CMD ["sh", "-c", "gunicorn src.inference:server --bind 0.0.0.0:7860 --workers 1 --threads 4 --timeout 120"]
