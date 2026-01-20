FROM ultralytics/ultralytics:latest-python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "src/inference.py"]