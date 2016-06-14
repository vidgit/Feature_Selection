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
#from nltk import PorterStemmer as stem

#Enter number of documents to be taken from corpus (<=100)
doc_num = [3204,1460,1398,1033]
# a=100
# doc_num = [a,a,a,a]

num_categories=len(doc_num)
# start = time.time()

#reading vocabulary

cwd = os.getcwd()
path = cwd + "/classicdocspreprocessed/terms.txt"
file  = open(path, 'r')
v = file.readlines()
file.close()

d=dict()
for item in v:
	vocabulary.append(item[:-1])
	d[item.rstrip('\n')]=0
	if item.rstrip('\n') == 'pseudotumor':
		print "8998"
# print vocabulary
# print term_freq
delimiters = " ", "-", "\n", "\r"
regexPattern = '|'.join(map(re.escape, delimiters))
# print regexPattern
print "reading documents"
#reading documents and applying stop words
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
# print trim("/avs//////")
for i in xrange(0, doc_num[0]):
	num = "%.6d" % (i+1)
	name = "/classic_30doc/cacm." + (str)(num)
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	z=[trim(x) for x in t if x != '']
	# print z
	# print z
	# for k in z:
	# 	# d[k]=d[k]+1
	# 	print k+" ",
	# print ""
	# 	print d[k]
	doc.append(z)
	# print doc
	orig_labels.append(1)
	file.close()
for i in xrange(0, doc_num[1]):
	num = "%.6d" % (i+1)
	name = "/classic_30doc/cisi." + (str)(num)
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	
	z=[trim(x) for x in t if x != '']
	# for k in z:
	# 	d[k]=d[k]+1
	doc.append(z)
	orig_labels.append(2)
	file.close()
for i in xrange(0, doc_num[2]):
	num = "%.6d" % (i+1)
	name = "/classic_30doc/cran." + (str)(num)
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	
	z=[trim(x) for x in t if x != '']
	# for k in z:
	# 	d[k]=d[k]+1
	doc.append(z)
	orig_labels.append(3)
	file.close()
for i in xrange(0, doc_num[3]):
	num = "%.6d" % (i+1)
	name = "/classic_30doc/med." + (str)(num)
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	
	z=[trim(x) for x in t if x != '' and x!='.']
	# for i in z:
	# 	print i+" ",
	# print ""
	# for k in z:
	# 	d[k]=d[k]+1
	doc.append(z)
	orig_labels.append(4)
	file.close()


s=set()
text_file = open("Output.txt", "w")
for j in doc:
	for i in j:
		s.add(i+"\n")
for j in s:
	text_file.write(j)
text_file.close()