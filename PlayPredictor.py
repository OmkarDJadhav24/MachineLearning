import pandas as pd
from sklearn.linear_model import LinearRegression
import csv
from sklearn.metrics import mean_squared_error

def main():
	#Data Load
	data=pd.read_csv('PlayPredictor.csv')

	print(data.head())

	# Data Cleaning
	data['Wether'].replace({'Sunny':0,'Overcast':1,'Rainy':2},inplace=True)
	data['Temperature'].replace({'Hot':0,'Mild':1,'Cool':2},inplace=True)
	data['Play'].replace({'No':0,'Yes':1},inplace=True)


	feature_data=data[['Wether','Temperature']]
	target_data=data[['Play']]

	# Training Data
	model=LinearRegression()
	model.fit(feature_data,target_data)

	# Testing Data
	prediction=model.predict(feature_data)

	# Score
	result=model.score(feature_data,target_data)
	print("Score of Data set is:",result)

if __name__=="__main__":
	main()