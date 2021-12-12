from vectorizer import fun
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import numpy

def classifiers(ngramVal):
    ret=[]
    emails_train,labels_train,emails_test,labels_test=fun(ngramVal)

    #Decision Tree Classifier
    classifier=tree.DecisionTreeClassifier().fit(emails_train,labels_train)
    val=classifier.predict(emails_test)
    ret.append(accuracy_score(val,labels_test))

    #Multi-Layer Peceptron Classifier
    classifier=MLPClassifier(random_state=1,max_iter=400).fit(emails_train,labels_train)
    val=classifier.predict(emails_test)
    ret.append(accuracy_score(val,labels_test))

    #Random Forest Classifier
    classifier=RandomForestClassifier(n_estimators=10,max_depth=1000).fit(emails_train,labels_train)
    val=classifier.predict(emails_test)
    ret.append(accuracy_score(val,labels_test))

    #KNN Classifier
    classifier=KNeighborsClassifier(n_neighbors=5,metric='euclidean').fit(emails_train,labels_train)
    val=classifier.predict(emails_test)
    ret.append(accuracy_score(val,labels_test))

    #SVM Classifier
    classifier=svm.SVC(kernel='linear', C=1,decision_function_shape='ovo').fit(emails_train, labels_train)
    val=classifier.predict(emails_test)
    ret.append(accuracy_score(val,labels_test))
    classifier=svm.SVC(kernel='rbf', gamma=1, C=1,decision_function_shape='ovo').fit(emails_train,labels_train)
    val=classifier.predict(emails_test)
    ret.append(accuracy_score(val,labels_test))
    classifier=svm.SVC(kernel='poly', degree=3, C=1,decision_function_shape='ovo').fit(emails_train,labels_train)
    val=classifier.predict(emails_test)
    ret.append(accuracy_score(val,labels_test))
    classifier=svm.SVC(kernel='sigmoid', C=1,decision_function_shape='ovo').fit(emails_train,labels_train)
    val=classifier.predict(emails_test)
    ret.append(accuracy_score(val,labels_test))

    return ret

def plotAccuracyDiffModels():
    print("Hello World")
    xLabel=['Decision Tree','MLP','Random Forest','KNN','SVM-Linear','SVM-Radial Basis Kernel','SVM-Polynomial','SVM-Sigmoid']
    fig,ax=plt.subplots()
    for i in range(1,4):
        yLabel=classifiers(ngramVal=i)
        ax.plot(xLabel,yLabel,marker="o")
        with open("CombinedResult.csv",'a') as csvfile:
            numpy.savetxt(csvfile,yLabel,delimiter=",")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Accuracy")
    ax.legend(['ngram value 1','ngram value 2','ngram value 3'])
    plt.xticks(rotation=45)
    plt.title("Different Models v/s Accuracy")
    plt.show()

def plotMLP():
    xLabel=[]
    for iterVal in range(1,1000,100):
        xLabel.append(iterVal)
    fig,ax=plt.subplots()
    for ngramVal in range(1,4):
        ret=[]
        emails_train,labels_train,emails_test,labels_test=fun(ngramVal)
        for i in range(1,1000,100):
            classifier=MLPClassifier(random_state=1,max_iter=i).fit(emails_train,labels_train)
            val=classifier.predict(emails_test)
            ret.append(accuracy_score(val,labels_test))
        with open("MLP.csv",'a') as csvfile:
            numpy.savetxt(csvfile,ret,delimiter=",")
        ax.plot(xLabel,ret,marker="o")
    ax.set_xlabel("Epoch Value")
    ax.set_ylabel("Accuracy")
    ax.legend(['ngram value 1','ngram value 2','ngram value 3'])
    plt.title("MaxIterations v/s Accuracy for MLP")
    plt.show()

def plotRandomForestClassifier():
    xLabel=[]
    for iterVal in range(1,100):
        xLabel.append(iterVal)
    fig,ax=plt.subplots()
    for ngramVal in range(1,4):
        ret=[]
        emails_train,labels_train,emails_test,labels_test=fun(ngramVal)
        for i in range(1,100):
            classifier=RandomForestClassifier(n_estimators=i,max_depth=1000).fit(emails_train,labels_train)
            val=classifier.predict(emails_test)
            ret.append(accuracy_score(val,labels_test))
        with open("RFC.csv",'a') as csvfile:
            numpy.savetxt(csvfile,ret,delimiter=",")
        ax.plot(xLabel,ret,marker="o")
    ax.set_xlabel("Number of Classifiers")
    ax.set_ylabel("Accuracy")
    ax.legend(['ngram value 1','ngram value 2','ngram value 3'])
    plt.title("Number of Classifiers v/s Accuracy for Random Forest")
    plt.show()

def plotKNN():
    xLabel=[]
    for iterVal in range(1,50):
        xLabel.append(iterVal)
    fig,ax=plt.subplots()
    for ngramVal in range(1,4):
        ret=[]
        emails_train,labels_train,emails_test,labels_test=fun(ngramVal)
        for i in range(1,50):
            classifier=KNeighborsClassifier(n_neighbors=i,metric='euclidean').fit(emails_train,labels_train)
            val=classifier.predict(emails_test)
            ret.append(accuracy_score(val,labels_test))
        with open("KNN.csv",'a') as csvfile:
            numpy.savetxt(csvfile,ret,delimiter=",")
        ax.plot(xLabel,ret,marker="o")
    ax.set_xlabel("Number of Nearest Neighbors")
    ax.set_ylabel("Accuracy")
    ax.legend(['ngram value 1','ngram value 2','ngram value 3'])
    plt.title("Number of Nearest Neighbors v/s Accuracy")
    plt.show()

def plotDTC():
    xLabel=[]
    for iterVal in range(1,50):
        xLabel.append(iterVal)
    fig,ax=plt.subplots()
    for ngramVal in range(1,4):
        ret=[]
        emails_train,labels_train,emails_test,labels_test=fun(ngramVal)
        for i in range(1,50):
            classifier=tree.DecisionTreeClassifier(max_depth=i).fit(emails_train,labels_train)
            val=classifier.predict(emails_test)
            ret.append(accuracy_score(val,labels_test))
        with open("DTC.csv",'a') as csvfile:
            numpy.savetxt(csvfile,ret,delimiter=",")
        ax.plot(xLabel,ret,marker="o")
    ax.set_xlabel("Maximum Depth")
    ax.set_ylabel("Accuracy")
    ax.legend(['ngram value 1','ngram value 2','ngram value 3'])
    plt.title("Maximum Depth v/s Accuracy for Decision Tree Classifier")
    plt.show()

plotMLP()