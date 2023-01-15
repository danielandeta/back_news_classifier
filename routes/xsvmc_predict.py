from fastapi import APIRouter
from models.schemas.xsvm_sch import XSVMC_BM
from  models.xsvmc_model.model import contextualized_prediction

app_xsvmc_predict_v1 = APIRouter()

@app_xsvmc_predict_v1.post('/xsvmc/predict',
                          tags=["Predictions"],
                          description="Get a classification from xSVMC")
async def get_prediction(xsvmc: XSVMC_BM):
    text = xsvmc.data
    predictions = contextualized_prediction(text)
    if (predictions):
        return {"predictions": predictions,
                "status": "ok"}
    return {"msg":"Error in predict "}
 