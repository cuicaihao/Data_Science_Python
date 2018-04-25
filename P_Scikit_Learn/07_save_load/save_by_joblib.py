from sklearn import svm
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
 
clf.fit(X, y)  

# save the model
joblib.dump(clf, 'model/svm_iris.pkl')

# load the model
clf2 = joblib.load('model/svm_iris.pkl')

y_pred = clf2.predict(X)

print('Training Acc of SVM on Iris: ', accuracy_score(y, y_pred))
 


 



