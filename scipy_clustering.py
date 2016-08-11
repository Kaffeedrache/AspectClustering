#!/usr/bin/env python
# encoding: utf-8

# (c) Wiltrud Kessler & N.D. (2015)
# You may do whatever you want with this piece of code.

"""
Clustering of words with Scipy hierarchical clustering.
"""

# See also:
# http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html#module-scipy.cluster.hierarchy
# http://www2.warwick.ac.uk/fac/sci/sbdtc/people/students/2010/jason_piper/code/


import numpy
import scipy.cluster.hierarchy as hac
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import sys

import distancemeasures  # our distance measures



if len(sys.argv) > 4:
   inputfilename = sys.argv[1]
   distancemeasure = sys.argv[2]
   clusterlinkage = sys.argv[3]
   clusternum = int(sys.argv[4])
   
else:
   print "Usage: python scipy_clustering.py <input file> <distance measure> <cluster linkage> <number of clusters>"
   print "Example: python scipy_clustering.py vectorexample.txt euclidean single 5"
   print ""
   print "   <input file> : file with one vector per line, first entry is the word, tab-separated"
   print "   <distance measure> : euclidean, cosine, levenshtein, levenshtein2, "
   print "         WordNet_path, WordNet_wup, WordNet_lch, WordNet_res, WordNet_lin, WordNet_jcn "
   print "         [see distancemeasures.py]"
   print "   <cluster linkage> : single, complete, average, weighted, centroid*, median*, ward*"
   print "         [see scipy documentation, methods with * can only be used with euclidean distance]"
   print "   <number of clusters> : an integer"
   sys.exit(1)


outputfilename = 'clustering_%s_%s_%d.txt' % (distancemeasure, clusterlinkage, clusternum)
print "Producing file: " + outputfilename


# Read the input file.
# Format: one vector per line, 
# first the word, then the vector entries, 
# separated by tab.
#     word \t number \t number ...
phrases = []
data = []
for line in open(inputfilename, 'r'):
   p = line.strip().split('\t')
   phrases.append(p[0])
   data.append(p[1:])


# Choose/Initialize distance measure (-> distancemeasures.py)
distances = distancemeasures.get_distance(data, phrases, distancemeasure)


# Initialize clusterer and do the clustering
# Hack: 'None' is returned by the distance measure if we select 'euclidean',
# because some linkage methods can only be used if we do not provide a matrix.
if distances == None:
   linkage = hac.linkage(data, method=clusterlinkage, metric='euclidean')
else:
   linkage = hac.linkage(distances, method=clusterlinkage)


# 'Flatten' the hierarchical clustering to get 'clusternum' clusters.
# flcuster gives the label for each data point in an array
#  -> convert to a dictionary for output
labels = hac.fcluster(linkage, clusternum,'maxclust')
clustdict = {i:[i] for i in xrange(len(linkage)+1)}
for i in xrange(len(linkage)-clusternum+1):
    clust1= int(linkage[i][0])
    clust2= int(linkage[i][1])
    clustdict[max(clustdict)+1] = clustdict[clust1] + clustdict[clust2]
    del clustdict[clust1], clustdict[clust2]
print clustdict


# Plot the dendrogram
plt.figure(figsize=(25,10))
plt.title("%s distance, %s linkage" % (distancemeasure, clusterlinkage))
plt.xlabel('words')
plt.ylabel('distance')
hac.dendrogram(
   linkage,
   leaf_rotation = 90.,
   leaf_font_size = 8.,
   labels = phrases, # show the words as labels for the instances
)
plt.show()


# Write the flattened clustering to the output file
outpufile = open(outputfilename, 'w')
outpufile.write("##Clustering result with distance measure=%s, cluster linkage=%s, number of clusters=%d\n" % (distancemeasure, clusterlinkage, clusternum))
i = 0
for val in clustdict.itervalues():
   i+=1
   print "-- cluster %d --" % (i)
   outpufile.write( "\n#--- cluster %d --\n" % (i))
   for line in val:
      print phrases[line]
      outpufile.write (phrases[line] + "\n")

outpufile.close()

