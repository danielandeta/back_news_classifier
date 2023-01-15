
from fastapi import APIRouter

from models.xsvmc_model.model import modelo, contextualized_prediction

app_home = APIRouter()
modelo= modelo()

@app_home.get('/', tags=["Intro"])
async def hello():
    return {"message": "Hello!"}


@app_home.get('/bye', tags=["Intro"])
async def bye():
    return {"message": "Bye!"}