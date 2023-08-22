import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
#------------------------------------------------

dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values
#-------------------------------------------------
total_acc =0
for i in range(0,20):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30)

    classifire = DecisionTreeClassifier(criterion="entropy")
    classifire.fit(x_train,y_train)
    y_pred = classifire.predict(x_test)
    #--------------------------------------------------------

    #Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    cs = confusion_matrix(y_test,y_pred)
    acc = (cs[0][0] + cs[1][1])/len(y_test)
    total_acc+=acc
    print("The Accurcy in : ",i,"Try = ",acc*100,"%")

print("Average Accuracy = ",(total_acc/20)*100,"%")