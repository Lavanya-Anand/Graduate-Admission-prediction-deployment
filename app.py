# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:03:18 2021

@author: Lavanya
"""

import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()


pickle_in = open("ridge_regression_best.pkl","rb")
model=pickle.load(pickle_in)

@app.get("/")
async def root():
   return {"message": "Hello World"}

@app.post("/predict")
def predict(TOFEL: float, LOR: float, CGPA:float, Research:str):
   if Research=='yes':
       x=1
   else:
       x=0
   x1=(TOFEL-90)/(120-90)
   x2=(LOR-1)/(5-1)
   x3=(CGPA-6)/(10-6)
   data=np.array([1,x1,x2,x3,x])
   prediction = model.predict(data)
   return {
       'prediction': prediction[0],
   }


if __name__ == '__main__':
   uvicorn.run(app, host='127.0.0.1', port=8000)
