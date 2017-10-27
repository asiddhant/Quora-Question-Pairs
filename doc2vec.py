import gensim.models as gnsm
from scipy.spatial import distance
import pandas as pd
import re

def main():
    # Loading the Model
    print 'Loading Model...'
    model = gnsm.Doc2Vec.load("C:/Users/ADITYA/Desktop/Quora/Doc2Vec/doc2vec.bin")
    print 'Model Load Complete...'
    
    # Loading the Data 
    print 'Loading Data...'
    quesData = pd.read_csv("allques.csv")
    print 'Data Load Complete...'
    
    start_alpha=0.01
    infer_epoch=1000
    print 'Calculating Vectors...'
    
    qvec=pd.DataFrame([["ID"]+["V"+ str(i) for i in range(1,301)]])
    qvec.to_csv('quesvectors.csv',header=False,index=False)

    for row in range(quesData.shape[0]):
        tqvec=model.infer_vector(re.sub(r'\W+', ' ', quesData.question[row]).lower().split(),alpha=start_alpha, steps=infer_epoch)
        qvec=pd.DataFrame([[quesData['id'][row]]+list(tqvec)])
        with open('quesvectors.csv', 'a') as f:
            (qvec).to_csv(f, header=False,index=False)
        if row%100==0:
            print (str(float(row)*100/float(quesData.shape[0]))+" % Complete")
    
    print 'Done... Terminating'
        
main()