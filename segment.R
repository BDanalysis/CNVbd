library(cghFLasso)
"segment" <- function(){
   path=getwd()
   data_path=paste(path,'RD',sep='/')
   data<-read.table(data_path)
   seg<-cghFLasso(data)
   segdata<-seg$Esti.CopyN
   out.file="seg.txt"
   write.table(segdata, file=out.file, sep="\t")

}
segment()
