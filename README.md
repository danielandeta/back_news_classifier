# back_news_classifier
Deployment of ML models using Python's Scikit-Learn + FastAPI 
# Dataset
## Spanish News Classification
For this workshop we are going to work with the following dataset: https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification
Or you can find the dataset in models/xsvmc_model/data 
# Python packages
The requirements.txt file should list all Python libraries that the project depend on, and they will be installed using:
```
  pip install -r requirements.txt
  ```
# Train
After we have install all the dependencies we can now run the script in models/xsvmc_model/model_train.ipynb, this script takes the input data and outputs a trained model.
# Web application
Finally we can test our web application by running:
  ```
  python -m uvicorn app:app --reload
  ```
