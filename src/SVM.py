import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def svm_classify(X_train, Y_train, X_test, Y_test):
    clf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='poly', degree=3, gamma='auto'))
    clf.fit(X_train, Y_train)
    clf.score(X_test,Y_test)
    predictions_NB = clf.predict(X_test)
    print("SVM Accuracy Score -> ",accuracy_score(predictions_NB, Y_test)*100)
    
def naive_classify(X_train, Y_train, X_test, Y_test):
    Naive = MultinomialNB()
    Naive.fit(X_train, Y_train)
    predictions_NB = Naive.predict(X_test)
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Y_test)*100)
    