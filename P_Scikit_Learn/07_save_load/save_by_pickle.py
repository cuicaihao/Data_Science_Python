from sklearn import svm
from sklearn import datasets
from sklearn.metrics import accuracy_score
import pickle

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
 
clf.fit(X, y)  

print('pickle save the model to string')
clf_string = pickle.dumps(clf)
print(type(clf_string))
clf_s = pickle.loads(clf_string)
y_pred = clf_s.predict(X)
print('Training Acc of SVM on Iris: ', accuracy_score(y, y_pred))
 

print('pickle save the model to binary')
clf_bin = pickle.dumps(clf, True)
print(type(clf_bin))
clf_b = pickle.loads(clf_string)
y_pred = clf_b.predict(X)
print('Training Acc of SVM on Iris: ', accuracy_score(y, y_pred))
 

print('pickle save the model to Disk:')
# save the model
with open('model/svm_iris.pickle', 'wb') as f:
	pickle.dump(clf, f)

# load the model
with open('model/svm_iris.pickle', 'rb') as f:
	clf_disk = pickle.load(f)
	y_pred = clf_disk.predict(X)
	print('Training Acc of SVM on Iris: ', accuracy_score(y, y_pred))
 


 



