import sys
import copy
from copy import deepcopy
import math  
import random
import random as randoma
import datetime
import numpy as np
import pandas as pd
import warnings
import pysam
from numpy import *
from numba import njit
from scipy import special
from scipy.stats import norm
from scipy.stats import rv_continuous, gamma
from sklearn import preprocessing
from scipy import stats
import configparser
config = configparser.ConfigParser()

#function to read fasta file
def readFasta(filename):
    seq = ''
    fread = open(filename)
    #delete row one
    fread.readline()

    line = fread.readline().strip()
    while line:
        seq += line
        line = fread.readline().strip()
        
    return seq


#function to read readCount file , generate readCount array
def readRd(filename, seqlen):
    print(seqlen)
    readCount = np.full(seqlen, 0.0)
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                posList = line.positions
                readCount[posList] += 1
        
    return readCount



def suspiciousCNVgain(binid,rd,binlength,normalRD,gainRD):
    rdA = np.mean(rd[binid * binlength: binid * binlength + binlength // 4])
    rdB = np.mean(rd[binid * binlength + binlength // 4: binid  * binlength+binlength // 2])
    rdC = np.mean(rd[binid * binlength + binlength // 2: binid  * binlength + (binlength // 4)*3])
    rdD = np.mean(rd[binid * binlength + (binlength // 4)*3: (binid+1) * binlength])
    print("rd",rdA,rdB,rdC,rdD)

def findborder(start,end,binlength,factor,rd,leftNoCnv,rightNoCnv,isgain,binnum):
    cnvAvgRD=np.mean(rd[start*binlength:(end)*binlength])
    if leftNoCnv>10:
        leftnormalRD=np.mean(rd[leftNoCnv*binlength:(leftNoCnv+10)*binlength])
    else:
        leftnormalRD = np.mean(rd[rightNoCnv * binlength:(rightNoCnv + 10) * binlength])
    if rightNoCnv<=binnum:
        rightnormalRD = np.mean(rd[rightNoCnv * binlength:(rightNoCnv + 10) * binlength])
    else:
        rightnormalRD = leftnormalRD
    leftbin = start-1
    rightbin = end+1
    if rightbin>binnum:
        rightbin=binnum
    print("start",start,"end",end,"LRD",leftnormalRD,"RRD",rightnormalRD,"CNVRD",cnvAvgRD)
    leftborder = findoneborderL(leftbin,start,binlength,factor,rd,leftnormalRD,cnvAvgRD,isgain)
    rightborder = findoneborderR(end-1, rightbin-1, binlength, factor, rd, rightnormalRD, cnvAvgRD,isgain)
    return leftborder,rightborder,start,end

def findoneborderL(lbin,rbin,binlength,factor,rd,LnormalRD,CNVRD,isgain):
    flagRD = (LnormalRD + CNVRD)/2 * 1
    print("flagRD",flagRD)
    if lbin==rbin : return lbin*factor
    #middle = (gainRD+normalRD)/2
    #print(lbin,rbin,binlength)
    # binlength=int(binlength)
    # lbin=int(lbin)
    # rbin=int(rbin)
    if factor%2==0:
        rdA = np.mean(rd[lbin * binlength: lbin * binlength + binlength//2])
        rdB = np.mean(rd[lbin * binlength + binlength//2: rbin * binlength])
        rdC = np.mean(rd[rbin * binlength: rbin * binlength + binlength//2])
        rdD = np.mean(rd[rbin * binlength + binlength//2: (rbin + 1) * binlength])
    else:
        a = int(lbin*factor)
        b = int(rbin*factor)
        c = int((rbin+1)*factor)
        if binlength>1:
            rdA = np.mean(rd[a: a + binlength//2])
            rdB = np.mean(rd[a + binlength//2: b])
            rdC = np.mean(rd[b: b + binlength//2])
            rdD = np.mean(rd[b + binlength//2: c])
        else:
            rdA = np.mean(rd[a: a + binlength])
            rdB = np.mean(rd[a + binlength: b])
            rdC = np.mean(rd[b: b + binlength])
            rdD = np.mean(rd[b + binlength: c])
    if np.isnan(rdA) or np.isnan(rdB) or np.isnan(rdC) or np.isnan(rdD):
        return lbin*factor
    print("rd", rdA, rdB, rdC, rdD)
    one = rdA-flagRD
    two = rdB-flagRD
    three = rdC - flagRD
    four = rdD - flagRD
    if isgain == 1:
        if(one <= 0 and two <=0 and three <=0 and four <=0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
            # return (rbin)*factor
        if(one > 0 and two > 0 and three > 0 and four > 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
            #return lbin*factor
        if(four > 0 and three <= 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if(three > 0 and two <= 0):
            return findoneborderL(2*lbin+1, 2*rbin, binlength//2,factor/2,rd,LnormalRD,CNVRD,isgain)
        if(two > 0 and one <= 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if(four <= 0 and three > 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if(three <= 0 and two > 0):
            return findoneborderL(2*lbin+1, 2*rbin, binlength//2,factor/2,rd,LnormalRD,CNVRD,isgain)
        if(two <= 0 and one > 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
    if isgain == 2:
        if(one <= 0 and two <=0 and three <=0 and four <=0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
            #return lbin*factor
        if(one > 0 and two > 0 and three > 0 and four > 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
            # return (rbin)*factor
        if(four <= 0 and three > 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if(three <= 0 and two > 0):
            return findoneborderL(2*lbin+1, 2*rbin, binlength//2,factor/2,rd,LnormalRD,CNVRD,isgain)
        if(two <= 0 and one > 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if(four > 0 and three <= 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if(three > 0 and two <= 0):
            return findoneborderL(2*lbin+1, 2*rbin, binlength//2,factor/2,rd,LnormalRD,CNVRD,isgain)
        if(two > 0 and one <= 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
            #return (rbin) * factor

def findoneborderR(lbin,rbin,binlength,factor,rd,RnormalRD,CNVRD,isgain):
    flagRD = (RnormalRD + CNVRD)/2 * 1
    print("flagRD", flagRD)
    if lbin==rbin : return lbin*factor
    #middle = (gainRD+normalRD)/2
    #print(lbin,rbin,binlength)
    # binlength=int(binlength)
    # lbin=int(lbin)
    # rbin=int(rbin)
    if factor%2==0:
        rdA = np.mean(rd[lbin * binlength: lbin * binlength + binlength//2])
        rdB = np.mean(rd[lbin * binlength + binlength//2: rbin * binlength])
        rdC = np.mean(rd[rbin * binlength: rbin * binlength + binlength//2])
        rdD = np.mean(rd[rbin * binlength + binlength//2: (rbin + 1) * binlength])
    else:
        a = int(lbin*factor)
        b = int(rbin*factor)
        c = int((rbin+1)*factor)
        if binlength>1:
            rdA = np.mean(rd[a: a + binlength//2])
            rdB = np.mean(rd[a + binlength//2: b])
            rdC = np.mean(rd[b: b + binlength//2])
            rdD = np.mean(rd[b + binlength//2: c])
        else:
            rdA = np.mean(rd[a: a + binlength])
            rdB = np.mean(rd[a + binlength: b])
            rdC = np.mean(rd[b: b + binlength])
            rdD = np.mean(rd[b + binlength: c])
    if np.isnan(rdA) or np.isnan(rdB) or np.isnan(rdC) or np.isnan(rdD):
        return rbin*factor
    print("rd", rdA, rdB, rdC, rdD)
    one = rdA-flagRD
    two = rdB-flagRD
    three = rdC - flagRD
    four = rdD - flagRD
    if isgain == 1:
        if(one <= 0 and two <=0 and three <=0 and four <=0):
            return findoneborderR(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
            # return lbin*factor
        if(one > 0 and two > 0 and three > 0 and four > 0):
            return findoneborderR(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
            # return (rbin)*factor
        if(one > 0 and two <= 0):
            return findoneborderR(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if(two > 0 and three <= 0):
            return findoneborderR(2*lbin+1, 2*rbin, binlength//2,factor/2,rd,RnormalRD,CNVRD,isgain)
        if(three > 0 and four <= 0):
            return findoneborderR(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if(one <= 0 and two > 0):
            return findoneborderR(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if(two <= 0 and three > 0):
            return findoneborderR(2*lbin+1, 2*rbin, binlength//2,factor/2,rd,RnormalRD,CNVRD,isgain)
        if(three <= 0 and four > 0):
            return findoneborderR(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
    if isgain == 2:
        if (one <= 0 and two <= 0 and three <= 0 and four <= 0):
            return findoneborderR(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
            # return lbin*factor
        if (one > 0 and two > 0 and three > 0 and four > 0):
            return findoneborderR(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
            # return (rbin)*factor
        if (one <= 0 and two > 0):
            return findoneborderR(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if (two <= 0 and three > 0):
            return findoneborderR(2 * lbin + 1, 2 * rbin, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if (three <= 0 and four > 0):
            return findoneborderR(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if (one > 0 and two <= 0):
            return findoneborderR(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if (two > 0 and three <= 0):
            return findoneborderR(2 * lbin + 1, 2 * rbin, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if (three > 0 and four <= 0):
            return findoneborderR(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)


if __name__ == '__main__':
    configfile = "config.ini"
    config.read(configfile)
    if config.has_option('OtherMethod', 'resultfile'):
        resultfile = config.get('OtherMethod', 'resultfile')
    else:
        raise Exception("No resultfile parameter")
    if config.has_option('OtherMethod', 'bamfile'):
        bam = config.get('OtherMethod', 'bamfile')
    else:
        raise Exception("No bamfilepath parameter")
    if config.has_option('OtherMethod', 'chrfile'):
        chrFile = config.get('OtherMethod', 'chrfile')
    else:
        raise Exception("No chrfile parameter")
    if config.has_option('BDS+Other', 'binLen'):
        binLen = config.getint('BDS+Other', 'binLen')
    else:
        binLen = 1000
    fread = open(resultfile)
    seq = readFasta(chrFile)
    # The length of seq
    seqlen = len(seq)
    binnum = int(seqlen/binLen)
    #f2read = open(GTfile)
    slong = 0
    sflag = 0
    start = 0
    now = 0
    end = 0
    type = 0
    CNVbin = []
    temnums = []
    CNVrangenum = []
    binlength = binLen
    while True:
        line = fread.readline()
        if not line:
            break
        strarr = line.split('\t')
        temnums.append(strarr)
    for index in range(len(temnums)):
        one = []
        one.append((int(temnums[index][0]) // binlength) - 10)  # LRD
        one.append((int(temnums[index][1]) // binlength) + 10)  # RRD
        one.append((int(temnums[index][0]) // binlength))  # startbin
        one.append((int(temnums[index][1]) // binlength))  # endbin
        one.append(temnums[index][3])
        CNVrangenum.append(one)
    print("result", CNVrangenum)
    rd = readRd(bam, seqlen)
    gainrandomnums = []
    lossrandomnums = []
    longtip = []
    resultbin =[]
    for j in CNVrangenum:
        nmrdnum=[]
        grdnum=[]
        lrdnum=[]
        leftbin = j[2]
        rightbin = j[3]
        startbin = j[2]
        endbin = j[3]
        cflag = 0

        if j[4]=='gain\n':
            oneresult = findborder(startbin,endbin,binlength,binlength,rd,j[0],j[1],1,binnum)
            if int(oneresult[1]) - int(oneresult[0])<0:
                continue
            one = []
            one.append(int(oneresult[0]))
            one.append(int(oneresult[1]))
            one.append(int(oneresult[1]) - int(oneresult[0]))
            one.append("gain")
            longtip.append(one)
            print("resultg",oneresult[0],oneresult[1])
            print("resultRD", np.mean(rd[int(oneresult[0]):int(oneresult[1])]))
            cflag=0
        if j[4] == 'loss\n':
            oneresult = findborder(startbin, endbin, binlength, binlength, rd, j[0], j[1],2,binnum)
            if int(oneresult[1]) - int(oneresult[0])<0:
                continue
            one = []
            one.append(int(oneresult[0]))
            one.append(int(oneresult[1]))
            one.append(int(oneresult[1]) - int(oneresult[0]))
            one.append("loss")
            longtip.append(one)
            print("resultg", oneresult[0], oneresult[1])
            print("resultRD", np.mean(rd[int(oneresult[0]):int(oneresult[1])]))
            cflag = 0
    borderresultfile = resultfile+"BDS.txt"
    output = open(borderresultfile, "w")
    for p in longtip:
        output.write(str(p[0]) + '\t' + str(p[1]) + '\t' + str(p[2]) + '\t' + str(p[3]) + '\n')



