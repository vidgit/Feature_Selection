import os
from nltk.tag import pos_tag
import numpy as np


cwd = os.getcwd()
path = cwd + "/classicdocspreprocessed/Output.txt"
file  = open(path, 'r')
v = file.readlines()
file.close()
# print "had"
d=dict()
c=0
b=set()
for item in v:
	item=item.rstrip('\n')
	print c
	b.add(item)
	# d[item]=pos_tag([item])[0]
	c=c+1
tagged=pos_tag(b)
# print tagged
for i in tagged:
	d[i[0]]=i

# print d
np.save('pos_tags.npy', d) 