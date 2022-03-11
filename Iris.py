import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from seaborn import countplot
from matplotlib.pyplot import figure,show
import numpy as np

def main():
	# Load Data
	load_data=load_iris()

	# Data cleaning
	feature_data=load_data.data
	target_data=load_data.target

	# Training Data
	train_data,test_data,train_target,test_target=train_test_split(feature_data,target_data,test_size=0.3)

	model=DecisionTreeClassifier()

	model.fit(train_data,train_target)

	# Testing Data
	prediction=model.predict(test_data)

	# Finding Accuracy
	Accuracy=accuracy_score(prediction,test_target)
	print("Accuracy of Testing is:",Accuracy*100)

if __name__=="__main__":
	main()