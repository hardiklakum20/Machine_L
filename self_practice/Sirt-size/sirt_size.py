import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv('D:\MCA Collage\Sem-3\Machine Learning\self_practice\Sirt-size\shirtsize.csv')
print(df.columns)
df



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['T Shirt Size'] = le.fit_transform(df['T Shirt Size'])
df

X = df.iloc[:, :-1].values
Y = df.iloc[:, 2].values
X
Y

#Split traing and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)

#K nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
#accuracies['KNN'] = acc
print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, Y_test)*100))
print(knn.score(X_test,Y_test))

#Predict for single
print(knn.predict([[158,58]]))
print(knn.predict([[170,68]]))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(Y_test, prediction)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, prediction)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,prediction)
print("Accuracy:",result2)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, Y_train)
prediction1 = clf.predict(X_test)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, Y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf.score(X_test, Y_test)))
result = confusion_matrix(Y_test, prediction1)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, prediction1)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,prediction1)
print("Accuracy:",result2)


#SVM
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, Y_train)
prediction2 = svm.predict(X_test)
print('Accuracy of SVM classifier on training set: {:.2f}'
      .format(svm.score(X_train, Y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
      .format(svm.score(X_test, Y_test)))
result = confusion_matrix(Y_test, prediction2)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, prediction2)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,prediction2)
print("Accuracy:",result2)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
prediction3 = gnb.predict(X_test)
print('Accuracy of GNB classifier on training set: {:.2f}'
      .format(gnb.score(X_train, Y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
      .format(gnb.score(X_test, Y_test)))
result = confusion_matrix(Y_test, prediction3)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, prediction3)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,prediction3)
print("Accuracy:",result2)


