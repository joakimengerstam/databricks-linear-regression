
# Use a lightweight Python base image
FROM python:3.10-slim

# Install curl for debugging
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . .

# Expose Flask port
EXPOSE 8080

# Environment variables (replace with real values or override at runtime)
ENV DATABRICKS_INSTANCE=https://dbc-a4864acb-d90e.cloud.databricks.com
ENV SERVING_ENDPOINT=nyc-taxi-fare
ENV DATABRICKS_TOKEN=dapi252f160d2af154aa8f13a166fbbcc55d

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
