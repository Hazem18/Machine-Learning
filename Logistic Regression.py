import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#--------------------------------------------
dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values
#---------------------------------------------
sum = 0 
l = 20
for i in range(0,l):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.25)

    #Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)


    #Fitting Logistic Regression
    classifire = LogisticRegression(random_state=0)
    classifire.fit(x_train,y_train)
    y_pred = classifire.predict(x_test)
    #-------------------------------------

    cm = confusion_matrix(y_test,y_pred)
    ac = (cm[0][0] + cm[1][1]) / len(y_test)
    sum+=ac

print("Average = ",(sum/l)*100)