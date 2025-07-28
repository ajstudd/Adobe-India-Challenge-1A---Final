FROM python:3.10

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install spaCy English model for POS features
RUN python -m spacy download en_core_web_sm

# Copy all necessary files and directories
COPY ./src ./src
COPY ./models ./models
COPY ./config_main.json .
COPY ./intelligent_filter.py .
COPY ./enhanced_metadata_extractor.py .
COPY ./generate_json_output.py .
COPY ./enhance_prediction_features.py .
COPY ./pos_features_handler.py .
COPY ./master_pipeline.py .

# Copy main pipeline script to root
COPY ./src/pipeline.py .

# Set environment variables
ENV MODE=1A
ENV USE_ML=true
ENV AUTOMATED_MODE=true
ENV PYTHONPATH=/app

# Create output directory
RUN mkdir -p /app/output

CMD ["python", "pipeline.py"]
