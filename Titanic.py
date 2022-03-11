import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.pyplot import figure,show
from seaborn import countplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
	# Load Data
	data=pd.read_csv('tested.csv')
	print(data.head())

	# Data Analyze
	print("Visualization:Survived or Non Servived")
	figure()
	target="Survived"

	countplot(data=data,x=target).set_title("Titanic:Survived or Non Survived")
	show()

	print("Visualization:Survived or Non Survived")
	figure()
	target="Survived"

	countplot(data=data,x=target,hue="Sex").set_title("Titanic:Survived or Non Survived")
	show()

	print("Visualization:Survived or Non Survived")
	figure()
	target="Survived"

	countplot(data=data,x=target,hue="Pclass").set_title("Titanic:Survived or Non Survived")
	show()

	data['Age'].plot.hist().set_title("Titanic:Survived or Non Survived")

	data['Fare'].plot.hist().set_title("Titanic:Survived and Non Survived")

	# Data Cleaning
	data.drop(["Name","PassengerId","SibSp","Parch","Ticket","Cabin","Embarked"],axis=1,inplace=True)

	data['Sex'].replace({'female':0,'male':1},inplace=True)
	print(data.head())

	feature_data=data[["Pclass","Sex","Age","Fare"]]
	target_data=data[["Survived"]]

	f_data=np.nan_to_num(feature_data)
	
	print(len(feature_data))
	print(len(target_data))
	print(len(f_data))

	train_data,test_data,train_target,test_target=train_test_split(f_data,target_data,test_size=0.5)

	# Data Training
	model=LogisticRegression()
	model.fit(train_data,train_target)

	# Data Testing
	prediction=model.predict(test_data)

	# Accuracy
	Class_report=classification_report(prediction,test_data)
	Confus_matrix=confusion_matrix(prediction,test_target)
	accuracy=accuracy_score(prediction,test_data)

	#print("Classification report:",Class_report)
	print("Confusion matrix:",Confus_matrix)
	print("Accuracy:",accuracy)


if __name__=="__main__":
	main()