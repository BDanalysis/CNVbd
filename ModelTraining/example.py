import numpy as np
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
import pickle

def loadDataSet():

    xList = []
    xLabels = []
    filename="/home/ywy/PycharmProjects/CNV-DPnetwork/software/release/ModelTraining/test.txt"
    fread = open(filename)
    # delete row one
    fread.readline()
    while True:
        line = fread.readline()
        if not line:
            break
        strarr = line.split('\t')
        nums=[]
        pos = []
        for k in range(1,6):
            if k<5:
                nums.append(float(strarr[k]))
            else:
                if(strarr[k]=="0\n"):
                    result = 0
                else:
                    if(strarr[k]=="1\n"):
                        result = 1
                    else:
                        result = 2
        if(nums[0]!=-10000):
            xList.append(nums)
            xLabels.append(result)
    return xList,xLabels


def GetResult():
    dataMat,labelMat=loadDataSet()
    clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(25,10), random_state=1,tol=1e-3,
                        learning_rate_init=1e-3,
                        #activation='relu'
                        )
    clf.fit(dataMat, labelMat)
    with open('testmodel.pickle', 'wb') as fw:
        pickle.dump(clf, fw)

if __name__=='__main__':
    GetResult()