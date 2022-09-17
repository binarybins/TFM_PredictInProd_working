FROM python:3.8.6-buster

COPY api /api
COPY TaxiFareModel /TaxiFareModel
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
COPY models /models
COPY data /data

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
