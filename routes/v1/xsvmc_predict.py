from fastapi import APIRouter
from models.schemas.xsvm_sch import XSVMC_BM, xSVMCResponse
import models.ml.classifier as clf

app_xsvmc_predict_v1 = APIRouter()


@app_xsvmc_predict_v1.post('/xsvmc/predict',
                          tags=["Predictions"],
                          response_model=xSVMCResponse,
                          description="Get a classification from xSVMC")
async def get_prediction(xsvmc: XSVMC_BM):
    data = dict(xsvmc)['data']
    prediction = clf.model.predict(data).tolist()
    probability = clf.model.predict_proba(data).tolist()
    log_probability = clf.model.predict_log_proba(data).tolist()
    return {"prediction": prediction,
            "probability": probability,
            "log_probability": log_probability}