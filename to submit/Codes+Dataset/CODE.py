from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn import tree,svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy
import csv

def extract_from_mail(filename):
    finalList=[]
    fields=[]
    with open(filename,'r',encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            finalList.append(row)
    emails=[]
    labels=[]
    for row in finalList:
        emails.append(row[0])
        labels.append(row[1])
    return emails,labels


def fun(ngramVal):

    #POPULATING EMAILS AND CORRESPONDING LABELS
    filename="2017B5A70828P.csv"
    emails,labels=extract_from_mail(filename)

    #splitting the data set into training and test data
    emails_train,emails_test,labels_train,labels_test = train_test_split(emails,labels,test_size=0.3,random_state=42)

    #transforming emails into word count
    vectorizer=TfidfVectorizer(ngram_range=(ngramVal,ngramVal),stop_words='english')
    emails_train_transformed=vectorizer.fit_transform(emails_train)
    emails_test_transformed=vectorizer.transform(emails_test)

    #selecting top 10% features else, the dimensionality will be very high
    selector = SelectPercentile(f_classif, percentile=20)
    selector.fit(emails_train_transformed,labels_train)
    emails_train_transformed=selector.transform(emails_train_transformed).toarray()
    emails_test_transformed=selector.transform(emails_test_transformed).toarray()
    return emails_train_transformed,labels_train,emails_test_transformed,labels_test

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
    #print("Hello World")
    xLabel=['Decision Tree','MLP','Random Forest','KNN','SVM-Linear','SVM-Radial Basis Kernel','SVM-Polynomial','SVM-Sigmoid']
    fig,ax=plt.subplots()
    for i in range(1,4):
        print("Running models for n-gram value "+str(i))
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

plotAccuracyDiffModels()