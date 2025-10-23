FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["python", "-m", "lerobot.cli", "deploy", "--ckpt", "outputs/checkpoint.pt", "--host", "0.0.0.0", "--port", "8000"]
