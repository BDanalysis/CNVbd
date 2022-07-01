"dc=0.5,prob=0.01"
"将rpy2那部分实现的功能，拆开进行编码"

import numpy as np
import pysam
import math
import sys
import matplotlib.pyplot as plt
from numba import njit
from sklearn.metrics import euclidean_distances
from scipy.stats import multivariate_normal
from sklearn import preprocessing
#import rpy2.robjects as robjects
import os
import pandas as pd
import datetime
import subprocess
#"/home/ywy/cghFLasso",repos = NULL, type="source"
def get_chrlist(filename):
    samfile = pysam.AlignmentFile(filename, "rb",ignore_truncation=True)
    List = samfile.references
    #print('List:',List)
    chrList = np.full(len(List), 0)
    for i in range(len(List)):
        chr = str(List[i]).strip('chr')
        if chr.isdigit():
            chrList[i] = int(chr)
    print('chrList',chrList)
    return chrList


def get_RC(filename, chrList, ReadCount,seqlen):
    samfile = pysam.AlignmentFile(filename, "rb",ignore_truncation=True)
    for line in samfile:
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                num = np.argwhere(chrList == int(chr))[0][0]#####################################33
                #print('num',num)
                posList = line.positions
                #print('posList',posList)	#
                for i in posList:
                    if(i<seqlen):
                        ReadCount[num][i] += 1
    print('ReadCount:',ReadCount)
    #print('lenposlist',len(posList))		#100
    return ReadCount


def read_ref_file(filename, ref, num):
    # read reference file
    if os.path.exists(filename):
        print("Read reference file: " + str(filename))
        with open(filename, 'r') as f:
            line = f.readline()
            for line in f:
                linestr = line.strip()
                ref[num] += linestr
    else:
        print("Warning: can not open " + str(filename) + '\n')
    return ref


def ReadDepth(ReadCount, binNum, ref,binSize):
    RD = np.full(binNum, 0.0)
    GC = np.full(binNum, 0)
    pos = np.arange(0, binNum)
    for i in range(binNum):
        RD[i] = np.mean(ReadCount[i * binSize:(i + 1) * binSize])
        cur_ref = ref[i * binSize:(i + 1) * binSize]
        N_count = cur_ref.count('N') + cur_ref.count('n')
        if N_count == 0:
            gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
        else:
            RD[i] = -10000
            gc_count = 0
        GC[i] = int(round(gc_count / binSize, 3) * 1000)

    index = RD > 0
    RD = RD[index]
    GC = GC[index]
    pos = pos[index]
    RD = gc_correct(RD, GC)

    return pos, RD


def gc_correct(RD, GC):
    # correcting gc bias
    bincount = np.bincount(GC)
    global_rd_ave = np.mean(RD)
    for i in range(len(RD)):
        if bincount[GC[i]] < 2:
            continue
        mean = np.mean(RD[GC == GC[i]])
        RD[i] = global_rd_ave * RD[i] / mean
    return RD


# def RDsimilar(RD,seg_rd):
#     simRD = np.full(len(RD), 0.0)
#     for k in range(1, len(RD)-1):
#         dot = 1 + abs(seg_rd[k+1] - seg_rd[k]) * abs(seg_rd[k] - seg_rd[k-1])
#         data = (1 + (pow(seg_rd[k+1] - seg_rd[k], 2))) * (1 + (pow(seg_rd[k] - seg_rd[k-1], 2)))
#         simRD[k] = dot/data
#     simRD[0] = simRD[1]
#     simRD[-1] = simRD[-2]
#     return simRD


def read_seg_file(filename):
    segRD = []
    with open(filename, 'r') as f:
        line = f.readline()
        for line in f:
            linestr = line.strip()
            linestrlist = linestr.split('\t')
            segRD.append(float(linestrlist[1]))
    segRD = np.array(segRD)
    return segRD

#没用到这个函数？？
def write(data, data1, data2):
    output = open("segrd.txt", "w")
    for i in range(len(data)):
        output.write(str(data[i]) + '\t' + str(data1[i]) + '\t' + str(data2[i]) + '\n')


def dis_matrix(RD):
    # calculating euclidean_distances matrix

    RD = RD.astype(np.float)
    pos = np.arange(0, len(RD))
    #pos = simRD.astype(np.float)
    nr_min = np.min(RD)
    nr_max = np.max(RD)
    newpos = (pos - min(pos)) / (max(pos) - min(pos)) * (nr_max - nr_min) + nr_min
    newpos = newpos.astype(np.float)

    RD = RD.astype(np.float)
    newpos = newpos.astype(np.float)
    rd = np.c_[RD, newpos]
    print("dis_matrix calculating")
    dis_matrix = euclidean_distances(rd, rd)
    return dis_matrix


def density(dis_matrix,dc):
    num = len(dis_matrix[0])
    ds = np.full(num, 0)
    for i in range(num):
        ds[i] = np.sum(dis_matrix[i] < dc)
    return ds


def distance(dis_matrix, ds):
    center = []
    num = len(dis_matrix[0])
    dt = np.full(num, 0.0)
    for i in range(num):
        index = ds > ds[i]
        if np.sum(index) == 0:
            center.append(i)
            dt[i] = np.max(dis_matrix[i])
        else:
            dt[i] = np.min(dis_matrix[i][index])#比该点密度大的点中，距离该点最近的点

    return dt, center


def get_dc_matrix(dis_matrix,dc):
    num = len(dis_matrix[0])
    dc_matrix = []
    pos = np.arange(num)
    for i in range(num):
        index = dis_matrix[i] < dc
        dc_matrix.append(pos[index])
    return dc_matrix


def read_RC_file(filename, data):
    with open(filename, 'r') as f:
        for line in f:
            linestr = line.strip()
            linestrlist = linestr.split('\t')
            pos = int(linestrlist[1])
            data[pos] = int(linestrlist[3])
    return data


def segment(pos, segrd):
    start = []
    end = []
    seg_rd = []
    i = 0
    j = 1
    while j < len(segrd):
        if j == len(segrd) - 1:
            start.append(int(pos[i]))
            end.append(int(pos[j-1]))
            seg_rd.append(float(segrd[i]))
            j += 1
        else:
            if segrd[i] == segrd[j]:
                j += 1
            else:
                start.append(int(pos[i]))
                end.append(int(pos[j-1]))
                seg_rd.append(float(segrd[i]))
                i = j
    return start, end, seg_rd


def write_CNV(filename, chr, start, end, type):
    count = 0
    output = open(filename, "w")
    for i in range(len(start)):
        if type[i] == 2:
            count += 1
            output.write('chr' + str(chr) + '\t' + str(start[i] + 1) + '\t' + str(end[i]) + '\t' + 'gain' + '\n')
        elif type[i] == 1:
            count += 1
            output.write('chr' + str(chr) + '\t' + str(start[i] + 1) + '\t' + str(end[i]) + '\t' + 'loss' + '\n')
    print(count)


def plot(pos, data):
    plt.scatter(pos, data, s=3, c="black")
    #plt.scatter(pos1, data1, s=3, c="red")

    max_pos=max(pos)
    max_data=max(data)
    plt.xlim(0,max_pos)
    plt.ylim(0,max_data)
    plt.xlabel("local density")
    plt.ylabel("minimum distance")
    plt.show()


def caculating_CNV(dc_m, center, ds, dt, start, end, rd,binSize):
    normRD = np.mean(rd[center])
    num = len(rd)
    flag = np.full(num, 0)
    CNV_start = []
    CNV_end = []
    CNV_type = []
    D_max_value = 0

    ds_score = preprocessing.scale(ds)
    dt_score = preprocessing.scale(dt)

    for i in range(len(center)):
        pos = dc_m[center[i]]
        flag[pos] = 1
        D_value = abs(rd[pos] - normRD)
        if max(D_value) > D_max_value:
            D_max_value = max(D_value)

    rd_value = abs(rd - normRD)
    mean_ds = np.mean(ds_score)
    mean_dt = np.mean(dt_score)
    mu = np.array([mean_ds, mean_dt])
    sigma = np.cov(ds_score, dt_score)
    var = multivariate_normal(mean=mu, cov=sigma)
    print("mu",mu)
    print('cov',sigma)
    for i in range(num):
        prob = var.pdf([ds_score[i], dt_score[i]])
        if prob < 0.01 and ds_score[i] < mean_ds:
            if rd[i] < normRD:
                type = 1
                CNV_type.append(int(1))
            else:
                type = 2
                CNV_type.append(int(2))
            CNV_start.append(start[i] * binSize)
            CNV_end.append(end[i] * binSize + binSize)
        #print(start[i] * binSize+ 1, end[i] * binSize + binSize, prob, ds_score[i], rd[i])

    CNVstart = np.array(CNV_start)
    CNVend = np.array(CNV_end)
    CNVtype = np.array(CNV_type)

    for i in range(len(CNVtype)-1):
        if CNVstart[i+1] <= CNVend[i] and CNVtype[i] == CNVtype[i+1]:
            CNVstart[i+1] = CNVstart[i]
            CNVtype[i] = 0
    index = CNVtype > 0
    CNVstart = CNVstart[index]
    CNVend = CNVend[index]
    CNVtype = CNVtype[index]

    return CNVstart, CNVend, CNVtype

def main(params):
    starttime = datetime.datetime.now()
    # get params
    #path = "/Volumes/TOSHIBA/NGS_data/SInC/10x/"
    #outpath = "/Volumes/TOSHIBA/NGS_data/SInC/10x/dp_result/"
    #bamName = sys.argv[1]
    #bam = path + bamName
    #bam = sys.argv[1]
    #bam = "sim34_6_6100_read.sort.bam" "/home/ywy/PycharmProjects/CNV-IFTV/software/data/sim" + str(num) + "_6_6100_read.sort.bam"
    #bam = "/home/ywy/PycharmProjects/CNV-IFTV/software/data/sim34_6_6100_read.sort.bam"
    binSize = params[0]
    chrnum = params[1]
    bam = params[2]
    dc = params[3]
    chrfile = params[4]
    #print(binSize,chrnum,bam,dc,chrfile)
    chrList = np.full(1, chrnum)
    chrNum = len(chrList)
    refList = [[] for i in range(chrNum)]
    #reflen = len(chrNum)
    for i in range(chrNum):
        reference = chrfile
        refList = read_ref_file(reference, refList, i)

    chrLen = np.full(chrNum, 0)

    for i in range(chrNum):
        chrLen[i] = len(refList[i])

    allds = np.full(int(chrLen[0] / binSize), -100.0)
    alldt = np.full(int(chrLen[0] / binSize), -100.0)
    print('base:',np.max(chrLen))
    print("Read bam file:", bam)
    ReadCount = np.full((chrNum, np.max(chrLen)), 0)
    ReadCount = get_RC(bam, chrList, ReadCount,np.max(chrLen))


    #plot(np.arange(ReadCount[0].size),ReadCount)
    for i in range(chrNum):
        binNum = int(chrLen[i]/binSize)+1
        pos, RD = ReadDepth(ReadCount[0], binNum, refList[i],binSize)
        with open('RD','w') as file:
            for c in range(len(RD)):
                file.write(str(RD[c]) + '\n')
    
        #plot(pos,RD)
        #v = robjects.FloatVector(RD)
        #m = robjects.r['matrix'](v, ncol=1)
        #robjects.r.source("segment.R")
        #robjects.r.segment(m)

        subprocess.call('Rscript segment.R',shell=True)
        segFile = "seg.txt"
        segRD = read_seg_file(segFile)
        start, end, seg_rd = segment(pos, segRD)
        #plot(np.arange(len(seg_rd)),seg_rd)

        start = np.array(start)
        end = np.array(end)
        seg_rd = np.array(seg_rd)
        #simRD = RDsimilar(seg_rd)
        dis_m = dis_matrix(seg_rd)
        print("calculate density")
        ds = density(dis_m,dc)
        print("calculate distance")
        dt, center = distance(dis_m, ds)
        print('lencenter:',len(center))
        ds_score = preprocessing.scale(ds)
        dt_score = preprocessing.scale(dt)

        for m in range(len(start)):
            for n in range(end[m]-start[m]+1):
                allds[start[m] + n] = ds_score[m]
                alldt[start[m] + n] = dt_score[m]
        return allds,alldt

