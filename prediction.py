import numpy as np
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
import pickle
import configparser
config = configparser.ConfigParser()

def loadDataSet(filename):
    yyList = []
    yyLabels = []
    pposnums = []
    #读取数据
    yList = []
    yLabels = []
    posnums = []
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
        for k in range(1,5):
            if k<5 :
                nums.append(float(strarr[k]))
        if nums[0]!=-10000 :
            yList.append(nums)
            posnums.append(strarr[0])
    yyLabels.append(yLabels)
    yyList.append(yList)
    pposnums.append(posnums)
    return yList,posnums


def GetResult():
    configfile = "config.ini"
    config.read(configfile)
    if config.has_option('Prediction','predictionfilepath'):
        filename = config.get('Prediction','predictionfilepath')
    else:
        raise Exception("No predictionfilepath parameter")
    if config.has_option('Prediction','model'):
        modelname = config.get('Prediction','model')
    else:
        raise Exception("No model")
    if config.has_option('Prediction','binLen'):
        binLen = config.getint('Prediction','binLen')
    else:
        binLen = 1500
    resultfileprefix = filename[0:len(filename)-14]
    print("resultfileprefix", resultfileprefix)
    print("filename", filename)
    preddataMat,pos=loadDataSet(filename)
    print("resultfileprefix",resultfileprefix)
    print("data finish")
    with open(modelname, 'rb') as fr:
        clf = pickle.load(fr)
    print("load finish")
    y_pred = clf.predict(preddataMat)
    m = len(y_pred)
    print(y_pred,m)
    savefile = []
    bordernums = []
    for i in range(m):
        oner = []
        oner.append(pos[i])
        oner.append(y_pred[i])
        savefile.append(oner)
        temborder=[]
        if y_pred[i]!=0:
            if y_pred[i] == 1:
                type = 'gain\n'
            if y_pred[i] == 2:
                type = 'loss\n'
            temborder.append((int(pos[i])) * binLen)
            temborder.append((int(pos[i]) + 1) * binLen)
            temborder.append(binLen)
            temborder.append(type)
            bordernums.append(temborder)
    saveresultfile = resultfileprefix+"_result-prediction.txt"
    saveresultfile1 = resultfileprefix+"_prediction-border.txt"
    print("aaa",saveresultfile)
    print("bbb", saveresultfile1)
    flag = 0
    bordernums1 = []
    for j in range(len(bordernums)):
        if flag == 0:
            flag = 1
            start = bordernums[j][0]
            end = bordernums[j][1]
            type = bordernums[j][3]
        if j < len(bordernums) - 1:
            if flag == 1:
                if end == bordernums[j + 1][0] and type == bordernums[j + 1][3]:
                    end = bordernums[j + 1][1]
                    continue
                else:
                    one = []
                    one.append(start)
                    one.append(end)
                    one.append(abs(start - end))
                    one.append(type)
                    bordernums1.append(one)
                    flag = 0
                    continue
        if j == len(bordernums) - 1:
            one = []
            one.append(start)
            one.append(end)
            one.append(abs(start - end))
            one.append(type)
            bordernums1.append(one)
    output = open(saveresultfile, "w")
    for i in savefile:
        output.write(str(i[0]) + '\t' + str(i[1]) +'\n' )
    output = open(saveresultfile1, "w")
    for i in bordernums1:
        output.write(str(i[0]) + '\t' + str(i[1]) + '\t'+ str(i[2]) + '\t' + str(i[3]))
    print("ccc")

if __name__=='__main__':
    GetResult()