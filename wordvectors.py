import numpy as np
import re
from collections import defaultdict

num_samples = 100000

left_seq_len = 10
right_seq_len = 10

embed_dimensions = 100

null_vec=[0]*embed_dimensions

glovedict=defaultdict(lambda:[0]*embed_dimensions)
glovefile='glove.6B.100d.txt'
with open(glovefile) as g:
	for line in g:
		tokens=line.strip().split(' ')
		vec=list(np.array(tokens[1:]).astype(float))
		keyd=tokens[0]
		glovedict[keyd]=vec

print 'Golve in Memory'

def tokenizeDoc(doc):
	return re.findall('\\w+',doc.lower())

full_data_file='quora_duplicate_questions.tsv'
outfile = open('full_data.csv', 'a+')
count=0
with open(full_data_file) as f:
	for line in f:
		if count>0 and count<=num_samples:
			tokens=line.strip().split('\t')
			label=int(tokens[5])
			ques1=tokenizeDoc(tokens[3])
			ques2=tokenizeDoc(tokens[4])
			ques1rep=[]
			ques2rep=[]
			for i in range(min(left_seq_len,len(ques1))):
				ques1rep+=glovedict[ques1[i]]

			for i in range(max(0,left_seq_len-len(ques1))):
				ques1rep+=null_vec

			for i in range(min(right_seq_len,len(ques2))):
				ques2rep+=glovedict[ques2[i]]

			for i in range(max(0,right_seq_len-len(ques2))):
				ques2rep+=null_vec

			resp=ques1rep+ques2rep+[label]
			resp=str(resp)
			outfile.write(resp[1:len(resp)-1]+'\n')

			if count%1000==0:
				print count

		count+=1

outfile.close()
