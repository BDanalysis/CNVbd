import EFiftv
import EFdpCNV
import configparser
import os
config = configparser.ConfigParser()

if __name__ == "__main__":
    configfile = "config.ini"
    config.read(configfile)
    if config.has_option('ExtractFeatures','bamfilepath'):
        bam = config.get('ExtractFeatures','bamfilepath')
    else:
        raise Exception("No bamfilepath parameter")
    if config.has_option('ExtractFeatures','chrfile'):
        chrFile = config.get('ExtractFeatures','chrfile')
    else:
        raise Exception("No chrfile parameter")
    if config.has_option('ExtractFeatures','binLen'):
        binLen = config.getint('ExtractFeatures','binLen')
    else:
        binLen = 1500
    if config.has_option('ExtractFeatures','dc'):
        dc = config.getfloat('ExtractFeatures','dc')
    else:
        dc = 0.5
    print(bam,chrFile,binLen,dc)
    treeNum = 256
    treeSampleNum = 256
    try:
        print("bam",bam)
        outputFile = bam + "_iftv_result.txt"
        paramsiftv = (binLen, chrFile, bam, treeNum, treeSampleNum)
        stdRDnums,CNVIFTVnums = EFiftv.main(paramsiftv)
        print("finish CNVIFTV")
        chrFile1 = os.path.basename(chrFile)
        chrDigit = chrFile1[3:len(chrFile1)-3]
        print("dsadsa",chrDigit)
        paramsdpcnv = (binLen, int(chrDigit), bam,dc,chrFile)
        allds,alldt=EFdpCNV.main(paramsdpcnv)
        print("finish CNVDP")
        outputFileUse = bam + "_parameter.txt"
        output = open(outputFileUse, "w")
        output.write('pos' + '\t' + 'RD' + '\t' + 'IFTV' + '\t' + 'DS' + '\t' + 'DT' + '\t' + 'ISCNV' + '\n')
        for j in range(len(stdRDnums)):
            output.write(str(j) + '\t' + str(stdRDnums[j]) + '\t' + str(CNVIFTVnums[j]) + '\t'
                         + str(allds[j]) + '\t' + str(alldt[j]) + '\n')
    except OSError:
        pass
