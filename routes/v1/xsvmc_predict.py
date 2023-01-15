from fastapi import APIRouter
from models.schemas.xsvm_sch import XSVMC_BM, xSVMCResponse
from  models.xsvmc_model.model import contextualized_prediction,predict_proba,predict_log_proba
app_xsvmc_predict_v1 = APIRouter()


@app_xsvmc_predict_v1.post('/xsvmc/predict',
                          tags=["Predictions"],

                          description="Get a classification from xSVMC")
async def get_prediction(xsvmc: XSVMC_BM):
    data = xsvmc.data
    prediction = contextualized_prediction(data)
    # probability = predict_proba(data).tolist()
    # log_probability = predict_log_proba(data).tolist()
    # return {"prediction": prediction,
    #         #"probability": probability,
    #         "log_probability": prediction}