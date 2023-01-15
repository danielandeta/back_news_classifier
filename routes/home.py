
from fastapi import APIRouter
from models.xsvmc_model.model import contextualized_prediction

app_home = APIRouter()

@app_home.get('/', tags=["Intro"])
async def hello():
    prueba = contextualized_prediction("Hola a todos")
    return {"message": prueba}


@app_home.get('/bye', tags=["Intro"])
async def bye():
    return {"message": "Bye!"}