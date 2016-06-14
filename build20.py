import os
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import heapq
import warnings
warnings.filterwarnings("ignore")
import time 
from stemming.porter2 import stem
from scipy import linalg, mat, dot
import timeit
from sklearn.feature_extraction.text import TfidfVectorizer
import gzip
import csv
from scipy.spatial.distance import correlation
import networkx as nx


#Enter number of documents to be taken from corpus (<=100)
#comp.graphics =973
#misc.forsale = 975
#rec.autos = 989
#sci.electronics = 984
#sci.med = 990
#sci.space = 987

doc_num = [973,975,989,984,990,987]
# a=100
# doc_num = [a,a,a,a]

# num_categories=len(doc_num)
# # start = time.time()

# #reading vocabulary
# vocabulary = []

cwd = os.getcwd()
# path = cwd + "/classicdocspreprocessed/terms.txt"
# file  = open(path, 'r')
# v = file.readlines()
# file.close()

d=dict()
delimiters = " ", "-", "\n", "\r", "\t"
regexPattern = '|'.join(map(re.escape, delimiters))
print "reading documents"
doc = []
orig_labels=[]
s=set(stopwords.words('english'))
st=list(string.punctuation)
def trim(x):
	x=x.lower()
	if x[0] in st:
		x=x[1:]
	if x=='':
		return x
	if x[-1] in st:
		x=x[0:-1]
	if x=='':
		return x
	if x[0] not in st and x[-1] not in st:
		return x
	else:
		return trim(x)
for i in xrange(0, doc_num[0]):
	name = "/20NG/comp.graphics/comp.graphics_" + (str)(i)+".txt"
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	z=[trim(x) for x in t if x != '']
	doc.append(z)
	orig_labels.append(1)
	file.close()

for i in xrange(0, doc_num[1]):
	name = "/20NG/misc.forsale/misc.forsale_" + (str)(i)+".txt"

	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))	
	z=[trim(x) for x in t if x != '']
	doc.append(z)
	orig_labels.append(2)
	file.close()

for i in xrange(0, doc_num[2]):
	name = "/20NG/rec.autos/rec.autos_" + (str)(i)+".txt"

	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))	
	z=[trim(x) for x in t if x != '']
	doc.append(z)
	orig_labels.append(3)
	file.close()

for i in xrange(0, doc_num[3]):
	name = "/20NG/sci.electronics/sci.electronics_" + (str)(i)+".txt"

	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	z=[trim(x) for x in t if x != '' and x!='.']
	doc.append(z)
	orig_labels.append(4)
	file.close()

for i in xrange(0, doc_num[4]):
	name = "/20NG/sci.med/sci.med_" + (str)(i)+".txt"

	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	z=[trim(x) for x in t if x != '' and x!='.']
	doc.append(z)
	orig_labels.append(5)
	file.close()

for i in xrange(0, doc_num[5]):
	name = "/20NG/sci.space/sci.space_" + (str)(i)+".txt"

	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	z=[trim(x) for x in t if x != '' and x!='.']
	doc.append(z)
	orig_labels.append(6)
	file.close()


s=set()
text_file = open(cwd+"/20NG/Output.txt", "w")
for j in doc:
	for i in j:
		s.add(i+"\n")
for j in s:
	text_file.write(j)
text_file.close()