import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_excel("Book1.xlsx")
x=df[["weight","height"]].values
y=df["class"].values
#trainning
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
#knn
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
nw=np.array([[57,170]])
y_pred=knn.predict(nw)
print(y_pred)
