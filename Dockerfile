FROM python:3.10

WORKDIR /app

COPY ./src ./src
COPY ./models ./models
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./src/pipeline.py .

CMD ["python", "pipeline.py"]
