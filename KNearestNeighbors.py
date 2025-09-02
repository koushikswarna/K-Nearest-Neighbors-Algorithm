import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('KNN_Project_Data')

print(df.head())

sns.pairplot(df,hue='TARGET CLASS')
plt.show()

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(df.drop('TARGET CLASS',axis=1))

scaled=scaler.transform(df.drop('TARGET CLASS',axis=1))

df1=pd.DataFrame(scaled,columns=df.columns[:-1])
df1.head()

from sklearn.model_selection import train_test_split

x=df1
y=df['TARGET CLASS']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

predictions=knn.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Choosing the Optimal K Value Using the Elbow Method

error_rate=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    predict=knn.predict(x_test)
    error_rate.append(np.mean(predict!=y_test))

sns.set_style('whitegrid')

sns.lineplot(x=range(1,40),y=error_rate,ls='dashed',color='blue',marker='o',markerfacecolor='red',markersize=10)
plt.show()

#Retain with Optimal K Value

knn=KNeighborsClassifier(n_neighbors=32)
knn.fit(x_train,y_train)
predictions=knn.predict(x_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))