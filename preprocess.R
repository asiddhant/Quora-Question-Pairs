dataset=read.csv("train.csv",stringsAsFactors = FALSE)

q1data=data.frame(id=dataset$qid1,question=dataset$question1)
q2data=data.frame(id=dataset$qid2,question=dataset$question2)
qdata=rbind(q1data,q2data)
rm(q1data,q2data)

qdata=qdata[-which(duplicated(qdata$id)),]
qdata=subset(qdata,qdata$question!="")
write.csv(qdata,"questions.csv",row.names=F)
