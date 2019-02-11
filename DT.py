#Michael Groff

import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import random
import time

def readin():
    #returns two dataframes with data to be used, last row being the result
    df1 = pd.read_csv('winequality-white.csv', sep=";")
    df2 = pd.read_csv('adult.txt', sep=",", header=None)
    df2.columns = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"]
    stacked = df2[["1","3","5","6","7","8","9","13","14"]].stack()
    df2[["1","3","5","6","7","8","9","13","14"]] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    return df1,df2

def split(df,size=1000,sizet=100):
    array = df.values
    m,n = array.shape
    ind = np.random.choice(m,size, replace = False)
    all = set(range(0,m))
    left = all - set(ind)
    left = list(left)
    m = len(left)
    indt = np.random.choice(m,sizet,replace = False)
    left = np.take(left,indt,axis=0)
    test = np.take(array,ind,axis=0)
    trial = np.take(array,left,axis=0)
    return test,trial

def prune(decisiontree, min_samples_leaf = 1):
    if decisiontree.min_samples_leaf >= min_samples_leaf:
        raise Exception('Tree already more pruned')
        #return
    else:
        decisiontree.min_samples_leaf = min_samples_leaf
        tree = decisiontree.tree_
        for i in range(tree.node_count):
            n_samples = tree.n_node_samples[i]
            if n_samples <= min_samples_leaf:
                tree.children_left[i]=-1
                tree.children_right[i]=-1

def cross(df, k=10):
    array = df.values
    m,n = array.shape
    size = int(m/k)
    tsize = m - (k-1)*size
    ind = np.random.choice(m,tsize, replace = False)
    test = np.take(array, ind, axis=0)
    all = set(range(0,m))
    left = all - set(ind)
    left = list(left)
    left = random.sample(left, len(left))
    train = np.take(array,left,axis=0)
    return train,test,size

def crossval(X,Y,size,pr = True,p=10,k=10):
    score = []
    for i in range(0,k-1):
        rem = range(size*i,size*i+size)
        rem = set(rem)
        m = X.shape[0]
        left = set(range(0,m)) - rem
        left = list(left)
        train = np.take(X,left,axis=0)
        tree = learn(train)
        if pr:
            prune(tree, p)
        a,b = test(tree,Y)
        c = accr(a,b)
        score.append(c)
    return score

def learn(array):
    X = array[:,:-2]
    Y = array[:,-1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,Y)
    return clf

def test(tree, array):
    X = array[:,:-2]
    Y = array[:,-1]
    Z = tree.predict(X)
    return Y,Z

def accr(a,b):
    c = np.where(a==b)
    c = np.asarray(c)
    k,tot = c.shape
    per = tot/a.size
    return per

def learningcurve(df,pr = True,p=15,n=100):
    m,z = df.shape
    size = int(0.7*m/n)
    sizes =[]
    traina = []
    testa = []
    times=[]
    for i in range(1,n):
        train,trial = split(df,size*i,int(0.3*m))
        s = time.clock()
        tree = learn(train)
        if pr:
            prune(tree,p)
        a,b = test(tree, trial)
        score = accr(a,b)
        c,d = test(tree, train)
        scoret = accr(c,d)
        e = time.clock()
        sizes.append(size*i)
        traina.append(scoret)
        testa.append(score)
        times.append(e-s)
    print
    print("Trial Times")
    print(times)
    return sizes,testa,traina

if __name__=="__main__":
    print ("Decision Tree")
    df1,df2 = readin()
    train1, trial1, size = cross(df1)
    score = crossval(train1,trial1,size)
    print("Cross Validation for Collection 1")
    print(score)
    m,n = df1.shape
    train1, trial1 = split(df1,int(0.7*m),int(0.3*m))
    s = time.clock()
    tree1 = learn(train1)
    prune(tree1,15)
    a,b = test(tree1, trial1)
    score = accr(a,b)
    c,d = test(tree1, train1)
    scoret = accr(c,d)
    e = time.clock()
    print("Testing Set Score for Collection 1")
    print(score)
    print("Training Set Score for Collection 1")
    print(scoret)
    print("Runtime")
    print(e-s)

    a,b,c = learningcurve(df1,50)
    plt.plot(a,b)
    plt.title("DT - Collection 1")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Accuracy")
    plt.legend(["Testing Sample"])
    plt.show()

    plt.plot(a,c)
    plt.title("DT - Collection 1")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Accuracy")
    plt.legend(["Training Sample"])
    plt.show()

    p = 50
    train1, trial1, size = cross(df2)
    score = crossval(train1,trial1,size,p)
    print("Cross Validation for Collection 2")
    print(score)
    m,n = df2.shape
    train1, trial1 = split(df2,int(0.7*m),int(0.3*m))
    s = time.clock()
    tree1 = learn(train1)
    prune(tree1,p)
    a,b = test(tree1, trial1)
    score = accr(a,b)
    c,d = test(tree1, train1)
    scoret = accr(c,d)
    e = time.clock()
    print("Testing Set Score for Collection 2")
    print(score)
    print("Training Set Score for Collection 2")
    print(scoret)
    print("Runtime")
    print(e-s)

    a,b,c = learningcurve(df2,p)
    plt.plot(a,b)
    plt.title("DT - Collection 2")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Accuracy")
    plt.legend(["Testing Sample"])
    plt.show()

    plt.plot(a,c)
    plt.title("DT - Collection 2")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Accuracy")
    plt.legend(["Training Sample"])
    plt.show()











    #footnote
