from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Knn_data():
	n=3
	# Load Data
	new_data=load_iris()

	feature_data=new_data.data
	target_data=new_data.target

	# Data Cleaning
	train_data,test_data,train_target,test_target=train_test_split(feature_data,target_data,test_size=0.5)

	#Training Data
	model=KNeighborsClassifier(n)
	model.fit(train_data,train_target)

	# Testing Data
	prediction=model.predict(test_data)

	# Accuracy
	Accuracy=accuracy_score(prediction,test_target)
	print("Accuracy of Testing is:",Accuracy*100,"%")

def main():
	Knn_data()

if __name__=="__main__":
	main()