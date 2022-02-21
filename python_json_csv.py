
import numpy as np
import os
import csv
import json
import pandas as pd

def MatGen(t, w):
    data = []
    arr = []
    
    for i in range(t):
        for j in range(w):
            arr.append(0)
            
        data.append(arr)
        arr = []
    
    return np.array(data)

def DataGen():
    file = r'facts.csv'
    df = pd.read_csv(file)
    questions = df['question']
    jsons = df['metadata']

    # Getting the list of raters
    ListOfRaters = []
    for items in jsons:
        outp = json.loads(items)    
        judg = outp["judgments"]
        for j in judg:
            rater= j['rater']
            if rater not in ListOfRaters:
                ListOfRaters.append(rater)

    workers = len(ListOfRaters)

    # Getting the list of questions
    ListOfQues = []
    for items in jsons:
        outp = json.loads(items) 
        ques = outp["question"]
        if ques not in ListOfQues:
            ListOfQues.append(ques)

    # Generating worker-task matrix
    questions = 250 # len(ListOfQues)
    wt = MatGen(questions, workers) # np.zeros(shape=(workers, questions))
    for q in range(0, questions):
        outp = json.loads(jsons[q])
        judgments = outp["judgments"]
        for j in judgments:
            rater=j['rater']
            ans=j['judgment']
            idx=ListOfRaters.index(rater)
            if(ans == 0):
                wt[q, idx] = 2
            else:
                wt[q, idx] = 1
    
    return wt

X = DataGen()
input(X.shape)
input(X[20:60,5:50])
    




