import nltk
from segment_tree import *
from sklearn.feature_extraction.text import TfidfVectorizer
from __future__ import unicode_literals, print_function
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import spacy
import re
from rouge import Rouge 
from collections import OrderedDict
import string
import numpy as np
from spacy.lang.en import English
import time
nl = English()
import sys
import pandas as pd
repeat = 5
data = []
doc = []
l3 = []
summary = []
hypothesis = ""
word_count = []
pair_similarity = []
summary_string = []
segtree_array = []

def count_word(index):
    global doc
    Doc = nl(doc[index])
    tokens = [t.text for t in Doc]
    tokens = [t for t in tokens if len(t.translate(t.maketrans('', '', string.punctuation + string.whitespace))) > 0] # + string.digits
    return len(tokens)

def store_word_count():
    global word_count,doc
    word_count = []
    for i in range(0,len(doc)):
        word_count.append(count_word(i))
        
def maximum(index, toPrint=0):
    global summary, pair_similarity
    length = len(summary)
    if(length!=0):
        max=0
        for i in range(length):
            a=pair_similarity[index][summary[i]]
            if(a>max):
                max=a
            if toPrint:
              print(str(summary[i])+" -> "+str(a))
        return max
    else:
        return 0

def count_sum(summary):
    sum=0
    length = len(summary)
    for i in range(length):
        sum+=count_word(summary[i])
    return sum

def mmr_sorted(lambda_, y, length):
    global word_count, pair_similarity, summary, segtree_array
    print('Inside MMR')
    l3 = []
    # vectorizer = TfidfVectorizer(smooth_idf=False)
    # X = vectorizer.fit_transform(doc)
    # y = X.toarray()
    rows = len(y)
    pair_similarity = []
    for i in range(rows):
        max=-1
        pair_similarity.append([])
        for j in range(rows):
            if(j!=i):
                a = np.sum(np.multiply(y[i],y[j]))
                pair_similarity[-1].append(a)
                if(a>max):
                    max=a
            else:
                pair_similarity[-1].append(1)
        l3.append(max)
    store_word_count()
    l = rows 
    count = 0
    last = -1
    summary = []
    summary_word_count = 0
    while(1):
        # print(summary_word_count)
        if (summary_word_count < length):
            max=-1
            for i in range(l):
                # a = maximum(i)
                a = segtree_array[i].query(0,rows-1,"max")
                mmrscore = lambda_*l3[i] - (1-lambda_)*a
                if(mmrscore >= max):
                    max = mmrscore
                    ind = i
            summary.append(ind)
            summary_word_count += word_count[ind]
            for i in range(l):
                segtree_array[i].update(ind,pair_similarity[i][ind])
        else:
            print('Bye')
            break

def listToString():  
    global summary_string, word_count, hypothesis, summary, doc
    summary_string = []
    leng = 0
    for i in summary:
      summary_string.append(doc[i])
      leng += word_count[i]
    hypothesis = "".join(summary_string) 

rouge_A1_avg = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]
rouge_A2_avg = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]

X1 = pd.read_csv('drive/My Drive/indian-summary-len-a1.txt', sep="\s", header=None)
X2 = pd.read_csv('drive/My Drive/indian-summary-len-a2.txt', sep="\s", header=None)

AllDocs = []
for i in range(0,50):
  with open(r'drive/My Drive/sentence_splitted/'+X1[0][i], 'r') as file:
    AllDocs.append(file.read().replace('\n',' '))
tf = TfidfVectorizer(smooth_idf=False)
X = tf.fit_transform(AllDocs)

rouge = Rouge()
sys.setrecursionlimit(429496729)
for i in range(0,50):
  length1=X1[1][i]
  length2=X2[1][i]
  doc = []
  y = []
  zeros = []
  segtree_array = []
  with open(r'drive/My Drive/sentence_splitted/'+X1[0][i], 'r') as file:
    for x in file:
      if x != '\n':
        doc.append(x)
        y.append(tf.transform([x]).toarray()[0])
        zeros.append(0)
  for i1 in range(len(y)):
    segtree_array.append(SegmentTree(zeros))
  with open(r'drive/My Drive/A1/'+X1[0][i],'r') as file:
    reference1 = file.read()

  with open(r'drive/My Drive/A2/'+X2[0][i], 'r') as file:
    reference2 = file.read()
  lamda=[0.3,0.5,0.7]
  for j in lamda:
    mmr_sorted(j,y,length1)
    listToString()
    print("Processing -> "+"sumA1_"+X1[0][i][:-4]+'_'+str(j)+".txt")
    f= open(r'drive/My Drive/A1sum/'+"sumA1_"+X1[0][i][:-4]+'_'+str(j)+".txt","w+")
    n = f.write(hypothesis)
    f.close()
    scores = rouge.get_scores(hypothesis, reference1)
    u = len(scores)
    for i1 in range(u):
      for k, v in scores[i1].items():
        for k_, v_ in v.items():
          rouge_A1_avg[i1][k][k_] += v_*0.02/3.0
    g= open("rogue.txt","a+")
    n = g.write(X1[0][i][:-4]+'\t'+"A1 - "+str(j)+'\t'+str(scores)+'\n')
    g.close()
    hypothesis=""
  for j in lamda:
    mmr_sorted(j,y,length2)
    listToString()
    print("Processing -> "+"sumA2_"+X2[0][i][:-4]+'_'+str(j)+".txt")
    f= open(r'drive/My Drive/A2sum/'+"sumA2_"+X2[0][i][:-4]+'_'+str(j)+".txt","w+")
    n = f.write(hypothesis)
    f.close()
    scores = rouge.get_scores(hypothesis, reference2)
    u = len(scores)
    for i1 in range(u):
      for k, v in scores[i1].items():
        for k_, v_ in v.items():
          rouge_A2_avg[i1][k][k_] += v_*0.02/3.0
    g= open("rogue.txt","a+")
    n = g.write(X2[0][i][:-4]+'\t'+"A2 - "+str(j)+'\t'+str(scores)+'\n')
    g.close()
print("Rouge A1 average :-")
print(rouge_A1_avg)
print("Rouge A2 average :-")
print(rouge_A2_avg)
