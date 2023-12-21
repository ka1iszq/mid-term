import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier ,plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
 
File_path = 'C:/Users/User/Downloads/'
File_name = 'car_data.csv'
 
df = pd.read_csv(File_path + File_name)
df.dropna(inplace=True)
encoder = LabelEncoder()
 
df = df.apply(encoder.fit_transform)
df.drop(columns=['User ID'], inplace=True)
 
 
 
model = DecisionTreeClassifier(criterion='entropy')
 
x_pred =['Male',46]
for i in range(0, len(df.columns)-1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
x_pred_adj = np.array(x_pred).reshape(-1,1)
 
x = df[['Gender','Age',]]
y = df['AnnualSalary']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=(0))
 
y_pred = model.printct(x_pred_adj)
print('Predition:',y_pred[0])
score = model.score(x, y)
print('Accuracy:','{:.2f}'.format(score))