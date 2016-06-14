from __future__ import division
import os
from nltk.corpus import stopwords
import copy
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from sklearn.feature_selection import chi2
import re
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time 
from stemming.porter2 import stem
from scipy import linalg, mat, dot
import timeit
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn import svm
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import f1_score
import random 
from collections import defaultdict
import operator
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from similarity import *
from ig import *

################################################################# READING DOCUMENTS ###################################################################################
total_time=time.time()
a1=973
b1=975
c1=989
d1=984
e1=990
f1=987
doc_num = [a1,b1,c1,d1,e1,f1]
doc_num=map(int, np.divide(doc_num,40))
print sum(doc_num),"Documents"
num_categories=len(doc_num)
total_time=time.time()
vocabulary = []

cwd = os.getcwd()
path = cwd + "/term20.txt"					# all possible terms are precomputed like in the moodle files
file  = open(path, 'r')							# this helps us quickly build a mapping for DOCUMENT FREQUENCIES
v = file.readlines()
file.close()

d=dict()
for item in v:
	vocabulary.append(item[:-1])
	d[item.rstrip('\n')]=0
	
delimiters = " ", "-", "\n", "\r","\t"
regexPattern = '|'.join(map(re.escape, delimiters))
print "Reading documents..."
################################################################## APPLYING STOPWORDS ###################################################################################
print "Applying stopwords..."
doc =[]
doc0=[]
orig_labels=[]
s=set(stopwords.words('english'))					# for stop word removal
st=string.punctuation
def trim(x):					# This function is called to remove punctuations
	x=x.lower()					# from either end of word '(abcd)' becomes 'abcd' The moodle code did not do this
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
for i in xrange(0, doc_num[0]):					# All documents are read separately
	name = "/20NG/comp.graphics/comp.graphics_" + (str)(i)+".txt"
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))	# Document is first split according to regex pattern
	z=[trim(x) for x in t if x != '']									#stop words are removed after taking lower case
	for k in z:															# Then unctuations are trimmed out
		d[k]=d[k]+1
	doc.append(z)
	doc0.append(z)
	orig_labels.append(1)
	file.close()
for i in xrange(0, doc_num[1]):
	name = "/20NG/misc.forsale/misc.forsale_" + (str)(i)+".txt"
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	
	z=[trim(x) for x in t if x != '']
	for k in z:
		d[k]=d[k]+1
	doc.append(z)
	doc0.append(z)
	orig_labels.append(2)
	file.close()
for i in xrange(0, doc_num[2]):
	name = "/20NG/rec.autos/rec.autos_" + (str)(i)+".txt"
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	
	z=[trim(x) for x in t if x != '']
	for k in z:
		d[k]=d[k]+1
	doc.append(z)
	doc0.append(z)
	orig_labels.append(3)
	file.close()
for i in xrange(0, doc_num[3]):
	name = "/20NG/sci.electronics/sci.electronics_" + (str)(i)+".txt"
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	
	z=[trim(x) for x in t if x != '']
	for k in z:
		d[k]=d[k]+1
	doc.append(z)
	doc0.append(z)
	orig_labels.append(4)
	file.close()
for i in xrange(0, doc_num[4]):
	name = "/20NG/sci.med/sci.med_" + (str)(i)+".txt"
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	
	z=[trim(x) for x in t if x != '']
	for k in z:
		d[k]=d[k]+1
	doc.append(z)
	doc0.append(z)
	orig_labels.append(5)
	file.close()
for i in xrange(0, doc_num[5]):
	name = "/20NG/sci.space/sci.space_" + (str)(i)+".txt"
	path = cwd + name
	file = open(path, 'r')
	t = file.read()
	t=filter(lambda w: not w.lower() in s,re.split(regexPattern, t))
	
	z=[trim(x) for x in t if x != '']
	for k in z:
		d[k]=d[k]+1
	doc.append(z)
	doc0.append(z)
	orig_labels.append(6)
	file.close()


######################## rarely occuring words #################################################
print "\nRemoving rarely occuring words"
c=set()             
doc1 = copy.deepcopy(doc)               # A paper referred to this technqiue as 'cut'
for i in xrange(len(doc)):              # words which occur a few times in whole corpus are removed 
    for j in xrange(len(doc[i])):
        if d[doc[i][j]] <= 6:
            doc1[i].remove(doc[i][j])
            c.add(doc[i][j])
        else:
            if doc[i][j] == '':
                doc1[i].remove('')
print (100*len(c))/float(len(d)), "percent features trimed by removing rarely occuring words"
##################################################################################################################

#########################   words with many meanings ###########################################################
print "\nRemoving elements with many meanings"
threshold = 11
pos_dict_words=set()                        # A paper found that words with many meanings dont contribute to classfication
moddoc =[]                                  # For example 'jaguar' may appear in two documents 
for i in xrange(0, len(doc1)):              # One document would we on wildlife, another on cars
    moddoc.append([])                       # When a document has many meany meanings it's liable to be used 
    for j in doc1[i]:                       # differently in the corpus
        if len(wn.synsets(j)) < threshold:              # wn.synsets(j) returns all meanings of that word
            moddoc[i].append(j)
            pos_dict_words.add(j)
                                            # Note: The paper asked for a benchmark of 5, but our experience says
pos_dict_words = list(pos_dict_words)       # a higher threshold is needed
read_ = pos_tag(pos_dict_words)
read_dictionary = dict()
for i in xrange(len(pos_dict_words)):
    read_dictionary[str(pos_dict_words[i])] = read_[i]
c0=set()
c2=set()
for i in moddoc:
    for j in i:
        c2.add(j)

for i in doc1:
    for j in i:
        c0.add(j)

print (100*(len(c0)-len(c2))/float(len(c0))), "percent features trimmed by removing words with many meanings"
doc1=copy.deepcopy(moddoc)
temp=len(pos_dict_words)
del moddoc
##################################################################################################################

#################################### Keeping only nouns and adjectives ##############################################

print "\nKeeping only nouns and adjectives"
doc2 = []                               # It has been observed that words which are not nouns contributed very little
for i in doc1:                          # 'very', 'fast', 'fifteen' etc generally dont tell us much about the class
    to_add = []                         # Terms like 'faith', 'police', etc do
    for k in i:                         # This hypothesis is valdiated in our work
        if read_dictionary[k][1][0] == 'N' or read_dictionary[k][1][0]=='J':
            to_add.append(k)
    doc2.append(to_add)                     # Note: Our group found that using adjectives also helps
doc = doc2                                  # This results in more words selected, but is a tradeoff which 
                                            # gives us more accuracy and f measure
# or read_dictionary[k][1][0]=='J'

for i in xrange(len(doc)) :
    if len(doc[i])==0:
        doc[i].append('and')
pos_dict_words=set()
for i in doc:
    for j in i:
        pos_dict_words.add(j)
pos_dict_words=list(pos_dict_words)
print (100*(temp-len(pos_dict_words)))/float(temp), "percent features trimed by removing only nouns and adjectives"
##################################################################################################################




############################## adopting similarity manoeuvres #################################################

print "\nUsing similarity measures"
# sim("man","woman") gives a high value
# sim("man","hospital") gives a low value
# On this basis we select words which are often used in similar contexts
# Papers have shown and our experiments have also shown that this works.
# Threshold is set manually
elem = []
count = 0
cp = dict()
for i in pos_dict_words:
    ch = set()
    for j in elem:
        # function call to another file below
        temp, temp1 = sim(i, j, lch_threshold=2.8) # Should be less than 4. Should be more than 1
        if temp == True:
            ch.add((temp1, j)) #This means the words i and j are similar, and can be considered for swapping
    if len(ch) == 0:    # If true, no match was found. The word is not replaced by any other word
        elem.append(i)  # Now future words will be compared with this word to see if THEY match with this word
        cp[i] = i
    else:
        cp[i] = max(ch)[1] # we select the word which matches with 'i' the most
    count = count + 1

#the loop below prints which words have been swapped with what
#this is for your convenience to see how threshold affects swapping
print "Original word : Swapped with"
for i in xrange(len(cp)):
    if pos_dict_words[i] != cp[pos_dict_words[i]]:
        print pos_dict_words[i], cp[pos_dict_words[i]]
cp['and']='and'
print (100 * (len(cp) - len(elem)))/len(cp), "percent features trimmed by similarity measure"
##################################################################################################################
##################################################################################################################
# Stemming the Documents, so as to compare with the words in vocabulary
print "\nStemming"
orig_labels0 = copy.deepcopy(orig_labels)
pos_dict_words_bns=set()
stemmed_doc = []
stemmed_doc0 = []
temp_doc = ""                               # Stemming is very standard, and stemming here mimics the moodle code file
temp_doc0 = ""                              # We additionally maintain stemmed_doc0 to keep track of the chi2 benchmark
for i in xrange(0, len(doc)):               
    stemmed_doc.append("")
    stemmed_doc0.append("")
    for j in xrange(len(doc[i])):
        temp_doc = temp_doc + stem(doc[i][j]) + " "
        pos_dict_words_bns.add(stem(doc[i][j]))
    # for item in doc0[i]:
    # 	temp_doc0=temp_doc0+stem(item)+" "
    for item in doc0[i]:
        temp_doc0 = temp_doc0 + stem(item) + " "

    stemmed_doc[i] = stemmed_doc[i] + temp_doc
    stemmed_doc0[i] = stemmed_doc0[i] + temp_doc0
    temp_doc = ""
    temp_doc0 = ""
pos_dict_words_bns=list(pos_dict_words_bns)

print "\nApplying tf-idf vectorization"
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(np.array(stemmed_doc))
vectors = vectors.todense()
vectors_np = np.array(vectors)
vectorizer = TfidfVectorizer()
vectors0 = vectorizer.fit_transform(np.array(stemmed_doc0))
vectors0 = vectors0.todense()
vectors_np0 = np.array(vectors0)

# vectors_np0 holds the benchmark chi2
# vectors_np holds the tfidf matrix for our work


################################################################ TF-BNS CALCULATOR ######################################################################################

tfdict=dict()
tfnum=0
for i in pos_dict_words_bns:
	tfdict[i]=tfnum
	tfnum = tfnum+1

# No. of rows = No. of docs
# No. od columns = No. of features

tfarray = np.zeros((len(doc),len(pos_dict_words_bns)))
for i in xrange(0,len(doc)):
	for j in stemmed_doc[i].split(" "):
		if j in tfdict:
			tfarray[i][tfdict[j]] = tfarray[i][tfdict[j]] + 1

# print tfarray
# print orig_labels

# TPR = No. of docs of class 1 which contain feature/ No. of docs of class 1
# FPR = No. of docs of not class 1 which contain feature / No. of docs of not class 1

bnsarray = np.zeros((len(pos_dict_words_bns),12))
tpr=0
fpr=0
bns=0
for i in xrange(0,len(pos_dict_words_bns)):
	for j in xrange(0,len(orig_labels)):
		if orig_labels[j]==1 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			tpr +=1
		if orig_labels[j]!=1 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			fpr +=1
	bnsarray[i][0]=tpr/a1
	if bnsarray[i][0] == 0:
		bnsarray[i][0]=0.0005
	
	bnsarray[i][1]=fpr/(b1+c1+d1+e1+f1)
	if bnsarray[i][1] == 0:
		bnsarray[i][1]=0.0005
	tpr=0
	fpr=0

for i in xrange(0,len(pos_dict_words_bns)):
	for j in xrange(0,len(orig_labels)):
		if orig_labels[j]==2 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			tpr +=1
		if orig_labels[j]!=2 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			fpr +=1
	bnsarray[i][2]=tpr/b1
	if bnsarray[i][2] == 0:
		bnsarray[i][2]=0.0005

	bnsarray[i][3]=fpr/(a1+c1+d1+e1+f1)
	if bnsarray[i][3] == 0:
		bnsarray[i][3]=0.0005
	tpr=0
	fpr=0

for i in xrange(0,len(pos_dict_words_bns)):
	for j in xrange(0,len(orig_labels)):
		if orig_labels[j]==3 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			tpr +=1
		if orig_labels[j]!=3 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			fpr +=1
	bnsarray[i][4]=tpr/c1
	if bnsarray[i][4] == 0:
		bnsarray[i][4]=0.0005
	bnsarray[i][5]=fpr/(a1+b1+d1+e1+f1)
	if bnsarray[i][5] == 0:
		bnsarray[i][5]=0.0005
	tpr=0
	fpr=0

for i in xrange(0,len(pos_dict_words_bns)):
	for j in xrange(0,len(orig_labels)):
		if orig_labels[j]==4 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			tpr +=1
		if orig_labels[j]!=4 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			fpr +=1
	bnsarray[i][6]=tpr/d1
	if bnsarray[i][6] == 0:
		bnsarray[i][6]=0.0005
	bnsarray[i][7]=fpr/(a1+b1+c1+e1+f1)
	if bnsarray[i][7] == 0:
		bnsarray[i][7]=0.0005
	tpr=0
	fpr=0

for i in xrange(0,len(pos_dict_words_bns)):
	for j in xrange(0,len(orig_labels)):
		if orig_labels[j]==5 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			tpr +=1
		if orig_labels[j]!=5 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			fpr +=1
	bnsarray[i][8]=tpr/e1
	if bnsarray[i][8] == 0:
		bnsarray[i][8]=0.0005
	bnsarray[i][9]=fpr/(a1+b1+c1+d1+f1)
	if bnsarray[i][9] == 0:
		bnsarray[i][9]=0.0005
	tpr=0
	fpr=0
for i in xrange(0,len(pos_dict_words_bns)):
	for j in xrange(0,len(orig_labels)):
		if orig_labels[j]==6 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			tpr +=1
		if orig_labels[j]!=6 and (pos_dict_words_bns[i] in stemmed_doc[j]):
			fpr +=1
	bnsarray[i][10]=tpr/f1
	if bnsarray[i][10] == 0:
		bnsarray[i][10]=0.0005
	bnsarray[i][11]=fpr/(a1+b1+c1+d1+e1)
	if bnsarray[i][11] == 0:
		bnsarray[i][11]=0.0005
	tpr=0
	fpr=0

finalbnsarray=np.zeros(len(pos_dict_words_bns))
for i in xrange(0,len(pos_dict_words_bns)):

	finalbnsarray[i]=(a1/sum(doc_num))*(norm.ppf(bnsarray[i][0])-norm.ppf(bnsarray[i][1]))+(b1/sum(doc_num))*(norm.ppf(bnsarray[i][2])-norm.ppf(bnsarray[i][3]))+(c1/sum(doc_num))*(norm.ppf(bnsarray[i][4])-norm.ppf(bnsarray[i][5]))+(d1/sum(doc_num))*(norm.ppf(bnsarray[i][6])-norm.ppf(bnsarray[i][7]))+(e1/sum(doc_num))*(norm.ppf(bnsarray[i][8])-norm.ppf(bnsarray[i][9]))+(f1/sum(doc_num))*(norm.ppf(bnsarray[i][10])-norm.ppf(bnsarray[i][11]))

finalbnsarray=np.absolute(finalbnsarray)
# print bnsarray
j=0
for i in xrange(0,len(finalbnsarray)):							# For the inf/NAN error
	if finalbnsarray[i]>1:
		finalbnsarray[i]=0.25

tfbnsmat = np.zeros((len(doc),len(pos_dict_words_bns)))
for i in xrange(len(doc)):
	for j in xrange(len(pos_dict_words_bns)):
		tfbnsmat[i][j]=tfarray[i][j]*finalbnsarray[j]
 

#####################################Chi Square###########################

# Note: chi2 is not implemented in the standard way
# We do not use SelectKBest, as the value of k is arbitrary and needs manual setting
# We select terms on basis of what their dependence is with the class label

def chi(vect,labels,benchmark=1):
    a, b = chi2(vect, labels)           # 'a' holds chi values, 'b' holds p values
    colnum = 0
    chivector = np.zeros((vect.shape[0], 1))
    for i in a:
        if i > benchmark:               # The term is selected only if it exceeds a benchmark  
            chivector = np.column_stack((chivector, vect[:, colnum]))
        colnum = colnum + 1             # The benchmark is not arbitrary
    chivector = chivector[:, 1:]        # A chi2 lookup table tells how much % independence is achieved
    return chivector                    # with what benchmark
chi2a=chi(vectors_np0,orig_labels0)
chi2_idf=chi(vectors_np,orig_labels)
chi2_bns=chi(tfbnsmat,orig_labels)


##############################Gini Index############################################
#Here, we have defined a function 'gini' which calculates the gini coeffecient value for a given matrix
#the list_of_values can be TF-IDF matrix or TF-BNS matrix or any other feature scoring matrix.
def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area
giniarray=np.zeros(1)
giniarraybns=np.zeros(1)
#Here, we are populating the giniarray and giniarraybns vector with the gini values for the given matrices
for i in range (0,vectors_np.shape[1]):
     giniarray=np.append(giniarray,gini(vectors_np[:,i]))
     giniarraybns=np.append(giniarraybns,gini(tfbnsmat[:,i]))

giniarray=giniarray[1:]
giniarraybns=giniarraybns[1:]
# print len(giniarray)
#This is the benchmark value below which we remove the feature. It generally lies between 0.85 and 0.95
#depending upon dataset
benchmarkgini=0.92
colnumgini=0
ginivector = np.zeros((vectors_np.shape[0],1))
for i in giniarray:
    if i > benchmarkgini:                               #keeping only features with gini>benchmark for idf
        ginivector=np.column_stack((ginivector,vectors_np[:,colnumgini]))
    colnumgini=colnumgini+1
ginivector = ginivector[:,1:]

colnumgini=0
ginivectorbns = np.zeros((tfbnsmat.shape[0],1))
for i in giniarraybns:
    if i > benchmarkgini:                               #keeping only features with gini>benchmark for bns
        ginivectorbns=np.column_stack((ginivectorbns,tfbnsmat[:,colnumgini]))
    colnumgini=colnumgini+1
ginivectorbns = ginivectorbns[:,1:]

###########################Gini Index Ends###############################################


########################### Splitting into train-test ###############################

toKeep = sorted(random.sample(range(0, len(vectors_np)), int(0.6 * len(vectors_np))))
def Split(vectors_, orig_labels_):
    global toKeep
    X_test = []
    X_train = []
    y_train = []
    y_test = []
    co = 0
    for x in xrange(len(vectors_)):
        if(co < len(toKeep)):
            if(x == toKeep[co]):
                X_train.append(vectors_[x].ravel())
                y_train.append(orig_labels_[x])
                co = co + 1
            else:
                X_test.append(vectors_[x].ravel())
                y_test.append(orig_labels_[x])
        else:
            X_test.append(vectors_[x].ravel())
            y_test.append(orig_labels_[x])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return (X_train, X_test, y_train, y_test)

ig_vector=ig_wrapper(vectors_np,orig_labels,0.1)
ig_vectorB=ig_wrapper(tfbnsmat,orig_labels,0.1)
    # The code for Information gain was soemwhat long, so it's kept in a different file.
    # It works on principles similar to the chi2 code
    # Please refer to ig.py for the same



X_train0, X_test0, y_train0, y_test0 = Split(chi2a, orig_labels0)
X_train, X_test, y_train, y_test = Split(chi2_idf, orig_labels)
X_trainB, X_testB, y_trainB, y_testB = Split(chi2_bns, orig_labels)
X_trainG, X_testG, y_trainG, y_testG = Split(ginivector,orig_labels)
X_trainI, X_testI, y_trainI, y_testI = Split(ig_vector,orig_labels)
X_trainIB, X_testIB, y_trainIB, y_testIB = Split(ig_vectorB,orig_labels)
X_trainGB, X_testGB, y_trainGB, y_testGB = Split(ginivectorbns,orig_labels)


##############################################################################################################


################################# Classification begins #############################################
def classifier(model,X,X1,y,y1):
    t0 = time.time()
    if model=='gnb':
        print 'GNB'
        gnb = GaussianNB().fit(X, y)
    elif model=='mnb':
        print 'MNB'
        gnb = MultinomialNB().fit(X,y)
    elif model=='bnb':
        print 'BNB'
        gnb = BernoulliNB().fit(X, y)
    elif model=='lin':
        print 'Linear SVM'
        gnb = svm.SVC(kernel='linear', C=0.5).fit(X, y)
    # elif model=='rbf':
    #     print 'RBF SVM'
    #     gnb = svm.SVC().fit(X, y)
    # elif model=='poly':
    #     print 'Poly SVM'
    #     gnb = svm.SVC(kernel='poly', degree=2).fit(X, y)
    elif model=='rfc':
        print 'Random Forest'
        gnb = RandomForestClassifier(max_depth=10, n_estimators=100, max_features=5).fit(X, y)
    elif model=='lr':
        print 'Logistic Regression'
        gnb = LogisticRegression().fit(X, y)
    elif model=='knn':
        print "K nearest neighbours"
        gnb = KNeighborsClassifier(n_neighbors=6).fit(X, y)
    y_pred = gnb.predict(X1)
    print accuracy_score(y1, y_pred), f1_score(y1, y_pred)
    print time.time() - t0

def Tester(model):
    global X_train, X_test, y_train, y_test, X_train0, X_test0, y_train0, y_test0,X_trainGB, X_testGB, y_trainGB, y_testGB
    global X_trainB, X_testB, y_trainB, y_testB,X_trainG, X_testG, y_trainG, y_testG,X_trainI, X_testI, y_trainI, y_testI 
    global X_trainIB, X_testIB, y_trainIB, y_testIB

    print "Chi Benchmark"
    classifier(model,X_train0, X_test0, y_train0, y_test0)

    print "Chi -tfidf"
    classifier(model,X_train, X_test, y_train, y_test)
    print "Chi -tfBns"
    classifier(model,X_trainB, X_testB, y_trainB, y_testB)

    print "Gini - idf"
    classifier(model,X_trainG,X_testG, y_trainG,y_testG)
    print "Gini - bns"
    classifier(model,X_trainGB,X_testGB, y_trainGB,y_testGB)
    
    print "IG -idf"
    classifier(model,X_trainI, X_testI, y_trainI, y_testI )
    print "IG - bns"
    classifier(model,X_trainIB, X_testIB, y_trainIB, y_testIB )
    
#############################################################################################################    



cases=['gnb','mnb','bnb','lin','rfc','lr','knn']
for i in cases:
    Tester(i)
    print "\n"

# print "finale"
# f=gzip.open("labels.csv.gz", "w")
# csv_w=csv.writer(f)
# le=vectors_np.shape[0]
# csv_w.writerow(orig_labels)
# f.close()

print "time =", time.time() - total_time
