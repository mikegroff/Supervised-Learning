#Michael Groff

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
import matplotlib.pyplot as plt
import random
import DT as dt
import time

def learn(array):
    X = array[:,:-2]
    Y = array[:,-1]
    clf = neighbors.KNeighborsClassifier(n_neighbors = 10)
    clf = clf.fit(X,Y)
    return clf

def crossval(X,Y,size,p=10,k=10):
    score = []
    for i in range(0,k-1):
        rem = range(size*i,size*i+size)
        rem = set(rem)
        m = X.shape[0]
        left = set(range(0,m)) - rem
        left = list(left)
        train = np.take(X,left,axis=0)
        tree = learn(train)
        a,b = dt.test(tree,Y)
        c = dt.accr(a,b)
        score.append(c)
    return score

def learningcurve(df,n=100):
    m,z = df.shape
    size = int(0.7*m/n)
    sizes =[]
    traina = []
    testa = []
    times = []
    for i in range(1,n):
        train,trial = dt.split(df,size*i,int(0.3*m))
        s = time.clock()
        tree = learn(train)
        a,b = dt.test(tree, trial)
        score = dt.accr(a,b)
        c,d = dt.test(tree, train)
        scoret = dt.accr(c,d)
        e = time.clock()
        sizes.append(size*i)
        traina.append(scoret)
        testa.append(score)
        times.append(e-s)
    print("Trial Times")
    print(times)
    return sizes,testa,traina

if __name__=="__main__":
    print ("KNN")
    df1,df2 = dt.readin()
    train1, trial1, size = dt.cross(df1)
    score = crossval(train1,trial1,size)
    print("Cross Validation for Collection 1")
    print(score)
    m,n = df1.shape
    train1, trial1 = dt.split(df1,int(0.7*m),int(0.3*m))
    s = time.clock()
    tree1 = learn(train1)
    #dt.prune(tree1,10)
    a,b = dt.test(tree1, trial1)
    score = dt.accr(a,b)

    c,d = dt.test(tree1, train1)
    scoret = dt.accr(c,d)
    e = time.clock()
    print("Testing Set Score for Collection 1")
    print(score)
    print("Training Set Score for Collection 1")
    print(scoret)
    print("Runtime")
    print(e-s)

    a,b,c = learningcurve(df1,100)
    plt.plot(a,b)
    plt.title("KNN - Collection 1")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Accuracy")
    plt.legend(["Testing Sample"])
    plt.show()

    plt.plot(a,c)
    plt.title("KNN - Collection 1")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Accuracy")
    plt.legend(["Training Sample"])
    plt.show()

    p = 50
    train1, trial1, size = dt.cross(df2)
    score = crossval(train1,trial1,size,p)
    print("Cross Validation for Collection 2")
    print(score)
    m,n = df2.shape
    train1, trial1 = dt.split(df2,int(0.7*m),int(0.3*m))
    s=time.clock()
    tree1 = learn(train1)
    #dt.prune(tree1,p)
    a,b = dt.test(tree1, trial1)
    score = dt.accr(a,b)

    c,d = dt.test(tree1, train1)
    scoret = dt.accr(c,d)
    e = time.clock()
    print("Testing Set Score for Collection 2")
    print(score)
    print("Training Set Score for Collection 2")
    print(scoret)
    print("Runtime")
    print(e-s)

    a,b,c = learningcurve(df2)
    plt.plot(a,b)
    plt.title("KNN - Collection 2")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Accuracy")
    plt.legend(["Testing Sample"])
    plt.show()

    plt.plot(a,c)
    plt.title("KNN - Collection 2")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Accuracy")
    plt.legend(["Training Sample"])
    plt.show()
