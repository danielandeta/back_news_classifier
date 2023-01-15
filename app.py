import models.xsvmc_model.classifier as clf
from fastapi import FastAPI
from joblib import load
from routes.v1.xsvmc_predict import app_xsvmc_predict_v1
from routes.home import app_home


app = FastAPI(title="XSVMC ML API", description="API for xsvmc ml model", version="1.0")


@app.on_event('startup')
async def load_model():
    clf.model = load('models/xsvmc_model/xsvmc.joblib')


app.include_router(app_home)
app.include_router(app_xsvmc_predict_v1, prefix='/v1')
