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
    samfile = pysam.AlignmentFile(filename, "rb", ignore_truncation=True)
    for line in samfile:
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                posList = line.positions
                for i in posList:
                    if (i < seqlen):
                        readCount[i] += 1

    return readCount



def suspiciousCNVgain(binid,rd,binlength,normalRD,gainRD):
    rdA = np.mean(rd[binid * binlength: binid * binlength + binlength // 4])
    rdB = np.mean(rd[binid * binlength + binlength // 4: binid  * binlength+binlength // 2])
    rdC = np.mean(rd[binid * binlength + binlength // 2: binid  * binlength + (binlength // 4)*3])
    rdD = np.mean(rd[binid * binlength + (binlength // 4)*3: (binid+1) * binlength])
    print("rd",rdA,rdB,rdC,rdD)

def findborder(start,end,binlength,factor,rd,leftNoCnv,rightNoCnv,isgain,binnum):
    cnvAvgRD=np.mean(rd[start*binlength:(end+1)*binlength])
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
    rightborder = findoneborderR(end, rightbin, binlength, factor, rd, rightnormalRD, cnvAvgRD,isgain)
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
        if (one <= 0 and two <= 0 and three <= 0 and four <= 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
            # return (rbin)*factor
        if (one > 0 and two > 0 and three > 0 and four > 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
            # return lbin*factor
        if (four > 0 and three <= 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if (three > 0 and two <= 0):
            return findoneborderL(2 * lbin + 1, 2 * rbin, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if (two > 0 and one <= 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if (four <= 0 and three > 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if (three <= 0 and two > 0):
            return findoneborderL(2 * lbin + 1, 2 * rbin, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if (two <= 0 and one > 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
    if isgain == 2:
        if (one <= 0 and two <= 0 and three <= 0 and four <= 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
            # return lbin*factor
        if (one > 0 and two > 0 and three > 0 and four > 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
            # return (rbin)*factor
        if (four <= 0 and three > 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if (three <= 0 and two > 0):
            return findoneborderL(2 * lbin + 1, 2 * rbin, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if (two <= 0 and one > 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if (four > 0 and three <= 0):
            return findoneborderL(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if (three > 0 and two <= 0):
            return findoneborderL(2 * lbin + 1, 2 * rbin, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
        if (two > 0 and one <= 0):
            return findoneborderL(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, LnormalRD, CNVRD, isgain)
            # return (rbin) * factor

def findoneborderR(lbin,rbin,binlength,factor,rd,RnormalRD,CNVRD,isgain):
    flagRD = (RnormalRD + CNVRD)/2 * 1
    print("flagRD", flagRD)
    if lbin==rbin : return lbin*factor
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
        if (one <= 0 and two <= 0 and three <= 0 and four <= 0):
            return findoneborderR(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
            # return lbin*factor
        if (one > 0 and two > 0 and three > 0 and four > 0):
            return findoneborderR(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
            # return (rbin)*factor
        if (one > 0 and two <= 0):
            return findoneborderR(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if (two > 0 and three <= 0):
            return findoneborderR(2 * lbin + 1, 2 * rbin, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if (three > 0 and four <= 0):
            return findoneborderR(2 * rbin, 2 * rbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if (one <= 0 and two > 0):
            return findoneborderR(2 * lbin, 2 * lbin + 1, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if (two <= 0 and three > 0):
            return findoneborderR(2 * lbin + 1, 2 * rbin, binlength // 2, factor / 2, rd, RnormalRD, CNVRD, isgain)
        if (three <= 0 and four > 0):
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
    if config.has_option('BDS','predictionresultfile'):
        filename = config.get('BDS','predictionresultfile')
    else:
        raise Exception("No predictionresultfile parameter")
    if config.has_option('BDS','bamfilepath'):
        bam = config.get('BDS','bamfilepath')
    else:
        raise Exception("No bamfilepath parameter")
    if config.has_option('BDS','chrfile'):
        chrFile = config.get('BDS','chrfile')
    else:
        raise Exception("No chrfile parameter")
    if config.has_option('BDS','binLen'):
        binLen = config.getint('BDS','binLen')
    else:
        binLen = 1500
    resultfileprefix = filename[0:len(filename)-22]
    fread = open(filename)
    seq = readFasta(chrFile)
    # The length of seq
    seqlen = len(seq)
    binnum = int(seqlen/binLen)
    slong = 0
    sflag = 0
    start = 0
    now = 0
    end = 0
    type = 0
    CNVbin = []
    CNVrangenum = []
    while True:
        line = fread.readline()
        if not line:
            break
        strarr = line.split('\t')
        if strarr[1]=="0\n":
            if sflag==0:
                continue
            else:
                slong+=1
                if slong == 10 or int(strarr[0])==binnum:
                    one = []
                    one.append(start)
                    if end > binnum: end = binnum
                    one.append(end)
                    one.append(CNVbin)
                    one.append(type)
                    CNVrangenum.append(one)
                    slong = 0
                    sflag = 0
                    start = 0
                    now = 0
                    end = 0
                    type = 0
                    CNVbin = []
                continue
        if sflag==0:
            start = int(strarr[0])-10
            now = int(strarr[0])
            end = int(strarr[0])+10
            type = int(strarr[1][0])
            sflag=1
            CNVbin.append(int(strarr[0]))
        else:
            if int(strarr[1][0]) != type:
                continue
            now = int(strarr[0])
            end = int(strarr[0])+10
            slong = 0
            CNVbin.append(int(strarr[0]))

    print("result", CNVrangenum)

    binlength = 1500
    rd = readRd(bam, seqlen)
    gainrandomnums = []
    lossrandomnums = []
    longtip = []
    resultbin =[]
    #print(rd)
    for j in CNVrangenum:
        if j[3]==1:
            for l in j[2]:
                gainrandomnums.append(l)
        else:
            for l in j[2]:
                lossrandomnums.append(l)
    print("aaaaanums",gainrandomnums,lossrandomnums)
    for j in CNVrangenum:
        leftbin = 0
        rightbin = 0
        startbin = 0
        endbin = 0
        cflag = 0
        # print("j[3]",j[3])
        if (((len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1)) < 1 and (
                len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1)) >= 0.2 and j[3] == 1) or
                ((len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1)) < 1 and (
                        len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1)) >= 0.4 and j[3] == 2)):
            cnvfactor = len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1)
            LNormalRD = np.mean(rd[j[0] * binlength:(j[0] + 10) * binlength])
            RNormalRD = np.mean(rd[(j[1] - 9) * binlength:j[1] * binlength])
            print("aaaaaaaaaaaaaaaaa", LNormalRD, RNormalRD)
            ccflag = 0
            temnums = j[2]
            leftborder = temnums[0]
            rightborder = temnums[len(temnums) - 1]
            temreass = []
            temreass2 = []
            print("oooooooooo", j[2], len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1))
            for kk in range(len(temnums)):
                if ccflag == 0:
                    startbin = temnums[kk]
                    endbin = temnums[kk]
                    ccflag = 1
                if kk < len(temnums) - 1 and temnums[kk + 1] == temnums[kk] + 1:
                    endbin = temnums[kk + 1]
                    continue
                else:
                    ccflag = 0
                    one = []
                    one.append(startbin)
                    one.append(endbin)
                    temreass.append(one)
                    temreass2.append(deepcopy(one))
            finalresult = temreass[0]
            # temreass2=copy.deepcopy(temreass)
            lastresult = []
            count = 0
            print("temreass", temreass, temreass2)
            for kkk in temreass:
                cnvRD = np.mean(rd[kkk[0] * binlength:(kkk[1] + 1) * binlength])
                reflagL = cnvfactor * LNormalRD + (1 - cnvfactor) * cnvRD
                reflagR = cnvfactor * RNormalRD + (1 - cnvfactor) * cnvRD
                print("LR", cnvRD, reflagL, reflagR)
                while True:
                    if j[3] == 1:
                        if kkk[0] > leftborder and np.mean(rd[(kkk[0] - 1) * binlength:(kkk[0]) * binlength]) > reflagL:
                            print("rdrdrd1", np.mean(rd[(kkk[0] - 1) * binlength:(kkk[0]) * binlength]))
                            kkk[0] = kkk[0] - 1
                            cnvRD = np.mean(rd[kkk[0] * binlength:(kkk[1] + 1) * binlength])
                        else:
                            break
                    if j[3] == 2:
                        if kkk[0] > leftborder and np.mean(rd[(kkk[0] - 1) * binlength:(kkk[0]) * binlength]) < reflagL:
                            print("rdrdrd1", np.mean(rd[(kkk[0] - 1) * binlength:(kkk[0]) * binlength]))
                            kkk[0] = kkk[0] - 1
                        else:
                            break
                    cnvRD = np.mean(rd[kkk[0] * binlength:(kkk[1] + 1) * binlength])
                    reflagL = cnvfactor * LNormalRD + (1 - cnvfactor) * cnvRD
                while True:
                    if j[3] == 1:
                        if kkk[1] < rightborder and np.mean(rd[kkk[1] * binlength:(kkk[1] + 1) * binlength]) > reflagR:
                            print("rdrdrd2", np.mean(rd[kkk[1] * binlength:(kkk[1] + 1) * binlength]))
                            kkk[1] = kkk[1] + 1
                        else:
                            break
                    if j[3] == 2:
                        if kkk[1] < rightborder and np.mean(rd[kkk[1] * binlength:(kkk[1] + 1) * binlength]) < reflagR:
                            print("rdrdrd2", np.mean(rd[kkk[1] * binlength:(kkk[1] + 1) * binlength]))
                            kkk[1] = kkk[1] + 1
                        else:
                            break
                    cnvRD = np.mean(rd[kkk[0] * binlength:(kkk[1] + 1) * binlength])
                    reflagR = cnvfactor * RNormalRD + (1 - cnvfactor) * cnvRD
                temresult = []
                print("kkk", kkk)
                print("temreass2", temreass2)
                for iii in range(len(temreass)):
                    if iii == count:
                        temresult.append(deepcopy(kkk))
                    else:
                        temresult.append(deepcopy(temreass2[iii]))
                count = count + 1
                print("temresult", temresult)
                lastresult.append(deepcopy(temresult))
            # finalresult = lastresult[0]
            print("lastresult", lastresult)
            Maxlength = 0
            oooresult = []
            for jjj in lastresult:
                print("jjj", jjj)
                length = 0
                for l in range(len(jjj)):
                    print("l", l)
                    startbinID = jjj[l][0]
                    while True:
                        oooresult.append(startbinID)
                        if startbinID != jjj[l][1]:
                            startbinID = startbinID + 1
                        else:
                            break
            print("oooresult", oooresult)
            setnew = set(oooresult)
            length = len(setnew)
            print("jdsiaojfsaafs", setnew, length)
            j[2] = list(setnew)
            j[2].sort()
            print("oooresult", setnew, j[2])

        if (((len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1)) < 1 and (
                len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1)) < 0.2 and j[3] == 1) or
                ((len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1)) < 1 and (
                        len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1)) < 0.4 and j[3] == 2)):
            cnvfactor = len(j[2]) / (j[2][len(j[2]) - 1] - j[2][0] + 1)
            LNormalRD = np.mean(rd[j[0] * binlength:(j[0] + 10) * binlength])
            RNormalRD = np.mean(rd[(j[1] - 9) * binlength:j[1] * binlength])
            print("aaaaaaaaaaaaaaaaa", LNormalRD, RNormalRD)
            ccflag = 0
            temdealnums = deepcopy(j[2])
            leftborder = temdealnums[0]
            rightborder = temdealnums[len(temdealnums) - 1]
            startindex = temdealnums[0]
            nowindex = 0
            temnums = []
            while nowindex < len(temdealnums):
                if startindex == temdealnums[nowindex]:
                    startindex += 1
                    nowindex += 1
                    continue;
                else:
                    temnums.append(startindex)
                    startindex += 1
            temreass = []
            temreass2 = []
            print("oooooooooo", temnums, cnvfactor)
            for kk in range(len(temnums)):
                if ccflag == 0:
                    startbin = temnums[kk]
                    endbin = temnums[kk]
                    ccflag = 1
                if kk < len(temnums) - 1 and temnums[kk + 1] == temnums[kk] + 1:
                    endbin = temnums[kk + 1]
                    continue
                else:
                    ccflag = 0
                    one = []
                    one.append(startbin)
                    one.append(endbin)
                    temreass.append(one)
                    temreass2.append(deepcopy(one))
            finalresult = temreass[0]
            # temreass2=copy.deepcopy(temreass)
            # prelastresult = []
            lastresult = []
            count = 0
            print("temreass", temreass, temreass2)
            for kkk in temreass:
                norRD = np.mean(rd[kkk[0] * binlength:(kkk[1] + 1) * binlength])
                reflagL = cnvfactor * LNormalRD + (1 - cnvfactor) * norRD
                reflagR = cnvfactor * RNormalRD + (1 - cnvfactor) * norRD
                print("LR", norRD, reflagL, reflagR)
                while True:
                    if j[3] == 1:
                        if kkk[0] > leftborder and np.mean(rd[(kkk[0] - 1) * binlength:(kkk[0]) * binlength]) < reflagL:
                            print("rdrdrd1", np.mean(rd[(kkk[0] - 1) * binlength:(kkk[0]) * binlength]))
                            kkk[0] = kkk[0] - 1
                        else:
                            break
                    if j[3] == 2:
                        if kkk[0] > leftborder and np.mean(rd[(kkk[0] - 1) * binlength:(kkk[0]) * binlength]) > reflagL:
                            print("rdrdrd1", np.mean(rd[(kkk[0] - 1) * binlength:(kkk[0]) * binlength]))
                            kkk[0] = kkk[0] - 1
                        else:
                            break
                    norRD = np.mean(rd[kkk[0] * binlength:(kkk[1] + 1) * binlength])
                    reflagL = cnvfactor * LNormalRD + (1 - cnvfactor) * norRD
                while True:
                    if j[3] == 1:
                        if kkk[1] < rightborder and np.mean(rd[kkk[1] * binlength:(kkk[1] + 1) * binlength]) < reflagR:
                            print("rdrdrd2", np.mean(rd[kkk[1] * binlength:(kkk[1] + 1) * binlength]))
                            kkk[1] = kkk[1] + 1
                        else:
                            break
                    if j[3] == 2:
                        if kkk[1] < rightborder and np.mean(rd[kkk[1] * binlength:(kkk[1] + 1) * binlength]) > reflagR:
                            print("rdrdrd2", np.mean(rd[kkk[1] * binlength:(kkk[1] + 1) * binlength]))
                            kkk[1] = kkk[1] + 1
                        else:
                            break
                    cnvRD = np.mean(rd[kkk[0] * binlength:(kkk[1] + 1) * binlength])
                    reflagR = cnvfactor * RNormalRD + (1 - cnvfactor) * cnvRD
                temresult = []
                print("kkk", kkk)
                print("temreass2", temreass2)
                for iii in range(len(temreass)):
                    if iii == count:
                        temresult.append(deepcopy(kkk))
                    else:
                        temresult.append(deepcopy(temreass2[iii]))
                count = count + 1
                print("temresult", temresult)
                lastresult.append(deepcopy(temresult))
            # finalresult = lastresult[0]
            print("lastresult", lastresult)
            Maxlength = 0
            oooresult = []
            for jjj in lastresult:
                print("jjj", jjj)
                length = 0
                for l in range(len(jjj)):
                    print("l", l)
                    startbinID = jjj[l][0]
                    while True:
                        oooresult.append(startbinID)
                        if startbinID != jjj[l][1]:
                            startbinID = startbinID + 1
                        else:
                            break
            print("oooresult", oooresult)
            setnew = set(oooresult)
            length = len(setnew)
            print("jdsiaojfsaafs", setnew, length)
            processnum = list(setnew)
            processnum.sort()
            startindex = temdealnums[0]
            endindex = temdealnums[len(temdealnums) - 1]
            endnum = []
            while startindex <= endindex:
                for num in processnum:
                    if startindex == num:
                        startindex += 1
                        continue
                endnum.append(startindex)
                startindex += 1
            print("endnum", endnum)
            print("j[2]", j[2])
            j[2] = endnum
            print("oooresult", j[2])

        if j[3]==1:

            for k in range(len(j[2])):
                binrd = np.mean(rd[j[2][k]*binlength:(j[2][k]+1)*binlength])
                print(j[2][k],binrd)
                if cflag==0:
                    startbin=j[2][k]
                    endbin=j[2][k]
                    cflag = 1
                if k < len(j[2])-1 and j[2][k+1]==j[2][k]+1:
                    endbin = j[2][k+1]
                    continue
                else:
                    cflag = 0
                if cflag == 0:
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
                    if j[2][k]<=oneresult[3]:
                        while True:
                            if k<len(j[2])-1:
                                k=k+1
                            else:
                                break
                            if j[2][k]>oneresult[3]:
                                k=k-1
                                break
        if j[3] == 2:
            for k in range(len(j[2])):
                binrd = np.mean(rd[j[2][k]*binlength:(j[2][k]+1)*binlength])
                print(j[2][k], binrd)
                if cflag == 0:
                    startbin = j[2][k]
                    endbin = j[2][k]
                    cflag = 1
                if k < len(j[2]) - 1 and j[2][k + 1] == j[2][k] + 1:
                    endbin = j[2][k + 1]
                    continue
                else:
                    cflag = 0
                if cflag == 0:
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
                    if j[2][k] <= oneresult[3]:
                        while True:
                            if k < len(j[2]) - 1:
                                k = k + 1
                            else:
                                break
                            if j[2][k] > oneresult[3]:
                                k = k - 1
                                break
    borderresultfile = resultfileprefix+"_result_border.txt"
    output = open(borderresultfile, "w")
    for p in longtip:
        output.write(str(p[0]) + '\t' + str(p[1]) + '\t' + str(p[2]) + '\t' + str(p[3]) + '\n')











