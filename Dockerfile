FROM ultralytics/ultralytics:latest-python

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files needed for install
COPY pyproject.toml ./
# If you have a lockfile, copy it too (recommended)
# COPY uv.lock ./

# Copy source code before editable install
COPY . .

# Install your package + deps into system (Render/Docker friendly)
RUN uv pip install --system -e .

EXPOSE 8050
CMD ["sh", "-c", "gunicorn src.inference:server --bind 0.0.0.0:${PORT:-8050} --workers 1 --threads 4 --timeout 120"]