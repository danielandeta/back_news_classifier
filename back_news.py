#run python -m uvicorn back_news:app --reload

# back_news.py https://engineering.rappi.com/using-fastapi-to-deploy-machine-learning-models-cd5ed7219ea
#$pydantic implementar validaciones de una forma muy sencilla.
#https://pydantic-docs.helpmanual.io/usage/models/
#https://www.geeksforgeeks.org/deploying-ml-models-as-api-using-fastapi/
"""
# Third party imports
from pydantic import BaseModel, Field

#from ms import app #https://pypi.org/project/ms/
#from ms.functions import get_model_response

# Model information
model_name = "xSVMC"
version = "v1.0.0"


# Input for data validation
class Input(BaseModel):
    noticia: str


# Ouput for data validation
class Output(BaseModel):
    categoria_1: str
    categoria_2: str
    categoria_3: str
    categorias: list() #otra opcion
    img_c1: str #i guess sería la ruta(?)
    img_c2: str
    img_c3: str


@app.get('/info')
async def model_info():
    #Return model information, version, how to call
    return {
        "name": model_name,
        "version": version
    }


@app.post('/predict', response_model=Output)
async def model_predict(input: Input):
    #Predict with input
    test_data = [[
            input
    ]]
    
     # Predicting the Class exampleee
    #class_idx = clf.predict(test_data)[0]
     
    # Return the Result exampleee
    # return { 'class' : iris.target_names[class_idx]}
"""

from fastapi import FastAPI
#import uvicorn 
from sklearn.datasets import load_iris
#from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel
# Creating FastAPI instance
app = FastAPI()
 
# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float
 
# Loading Iris Dataset
iris = load_iris()
 
# Getting our Features and Targets
X = iris.data
Y = iris.target
 
# Creating and Fitting our Model
clf = GaussianNB()
clf.fit(X,Y)
 
# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict')
def predict(data : request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
    ]]
     
    # Predicting the Class
    class_idx = clf.predict(test_data)[0]
     
    # Return the Result
    return { 'class' : iris.target_names[class_idx]}