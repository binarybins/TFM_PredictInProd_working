
# $DELETE_BEGIN
from datetime import datetime
import pytz

import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2


@app.get("/")
def index():
    return dict(greeting="hello")


@app.get("/predict")
def predict(pickup_datetime,        # 2013-07-06 17:18:00
            pickup_longitude,       # -73.950655
            pickup_latitude,        # 40.783282
            dropoff_longitude,      # -73.984365
            dropoff_latitude,       # 40.769802
            passenger_count):       # 1

    # create datetime object from user provided date
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user provided datetime with the NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # convert the user datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    # format the datetime as expected by the pipeline
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    # fixing a value for the key, unused by the model
    # in the future the key might be removed from the pipeline input
    # eventhough it is used as a parameter for the Kaggle submission
    key = "2013-07-06 17:18:00.000000119"

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
        key=[key],
        pickup_datetime=[formatted_pickup_datetime],
        pickup_longitude=[float(pickup_longitude)],
        pickup_latitude=[float(pickup_latitude)],
        dropoff_longitude=[float(dropoff_longitude)],
        dropoff_latitude=[float(dropoff_latitude)],
        passenger_count=[int(passenger_count)]))

    # ⚠️ TODO: get model from GCP

    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(fare=pred)
# $DELETE_END
