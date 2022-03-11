from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Classi(weight,surface):
	BallFeatures=[[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
	
	names=[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

	model=DecisionTreeClassifier()

	model.fit(BallFeatures,names)

	result=model.predict([[weight,surface]])

	if result==1:
		print("This is Tennis Ball")
	elif result==2:
		print("This is Cricket Ball")

def main():
	print("Enter the Weight of object:")
	weight=input()

	print("Enter the surface of object:")
	surface=input()

	if surface.lower=="rough":
		surface=1
	elif surface.lower=="smooth":
		surface=0

	Classi(weight,surface)

if __name__=="__main__":
	main()