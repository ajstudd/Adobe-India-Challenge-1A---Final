FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy all necessary files and directories
COPY ./src ./src
COPY ./models ./models
COPY ./config_main.json .
COPY ./generate_json_output.py .
COPY ./intelligent_filter.py .
COPY ./enhanced_metadata_extractor.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the entry point to our automated script
CMD ["python", "generate_json_output.py"]
