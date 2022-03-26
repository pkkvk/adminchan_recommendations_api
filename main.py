import fastapi
import xgboost as xg
from fastapi import Request
import pickle
import pandas as pd


#Load regression model
xgb_model_loaded = xg.XGBRegressor()
xgb_model_loaded.load_model("xgb_reg.json")

college_data = pd.read_csv("ForbesAmericasTopColleges2019.csv")

total = len(college_data)
# print(total)

increment = (5-1)/total
# print(increment)

college_list ={}
rating = 5.0
for i,value in college_data.iterrows():
    college_list[rating] = value["Name"]
    rating = rating - increment

# print(college_list)

# app= fastapi.App()

# @app.get("/recommendations")
# async def get_recommendations(request:Request,GRE_score : int,TOEFL_score : int,\
#     SOL_score : float,LOR_score : float,cgpa : float,research: bool):
#     data = calculate_recommendations(GRE_score,TOEFL_score,SOL_score,LOR_score,cgpa,research)
#     if data is not None:
#         return data
#     else:
#         return {}

def calculate_recommendations(GRE_score : int,TOEFL_score : int,\
    SOL_score : float,LOR_score : float,cgpa : float,research: bool):
    y=[]
    for rating,college in college_list.items():
        y.append([GRE_score,TOEFL_score,rating,SOL_score,LOR_score,cgpa,research])
    data=pd.DataFrame(y,columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'])
    pred = xgb_model_loaded.predict(data)
    data["Chance of Admit"]=pred
    count=0
    recommendations = {}
    for i,value in data.sort_values("Chance of Admit").iterrows():
        recommendations[college_list[value["University Rating"]]]=value["Chance of Admit"]*100
        count+=1
        if count==10:
            break
    print(recommendations)


calculate_recommendations(300,100,4.6,4.6,9.6,1)





