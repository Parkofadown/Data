import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('diabetes.csv')

dataset.head(5)

dataset.shape

dataset.describe()

corr=dataset.corr()
fig,ax=plt.subplots(figsize=(13,13))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)),corr.columns)
plt.yticks(range(len(corr.columns)),corr.columns)

features=dataset.drop(['Outcome'],axis=1)
labels=dataset['Outcome']

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.25)

from sklearn.neighbors import KNeighborsClassifier
classifer=KNeighborsClassifier()

classifer.fit(features_train,labels_train)

pred=classifer.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(labels_test,pred)
print('Accuracy:{}'.format(accuracy))
