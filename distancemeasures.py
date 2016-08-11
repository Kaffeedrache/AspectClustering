#!/usr/bin/env python
# encoding: utf-8

# (c) Wiltrud Kessler & N.D. (2015)
# You may do whatever you want with this piece of code.

"""
Distance measures for Scipy hierarchical clustering.
"""

import distance # levenshtein distance
from scipy import spatial # euclidean, cosine distance
import numpy #  euclidean, cosine distance

from nltk import TreebankWordTokenizer # tokenization of phrases for WordNet distances
from nltk.corpus import wordnet as wn # WordNet distances
from itertools import product # WordNet distances
from nltk.corpus import wordnet_ic # WordNet distances
brown_ic = wordnet_ic.ic('ic-brown.dat')  # WordNet_res



def getHead (word):
   """
   Get the head word of a phrase, 
   current approximation: Last token of a phrase.
   """
   tokens = TreebankWordTokenizer().tokenize(word)
   lasttoken = tokens[len(tokens)-1]
   return lasttoken


def getAllSynsets (word):
   """
   Get all WordNet synsets of all words of a phrase.
   """
   tokens = TreebankWordTokenizer().tokenize(word)
   ss = []
   for word in tokens:
      ss.extend(wn.synsets(word))
   return ss



def getWNSimilarity (synsetdotproduct, method):
   """
   Calculate the similarity between all pairs of synsets given in 'synsetdotproduct'
   with the given method.
   """
   try:
      if method == "WordNet_path": # 0-1
         similarity = max(s1.path_similarity(s2) for (s1,s2) in synsetdotproduct)         
      elif method == "WordNet_wup": # 0-1
         similarity = max(s1.wup_similarity(s2) for (s1, s2) in synsetdotproduct)
      elif method == "WordNet_lch": # nicht 0-1
         similarity = max(s1.lch_similarity(s2) for (s1, s2) in  synsetdotproduct)
      elif method == "WordNet_res": # nicht 0-1
         simslist = []
         for (s1, s2) in synsetdotproduct:
            try:
               simslist.append(s1.res_similarity(s2, brown_ic))
            except nltk.corpus.reader.wordnet.WordNetError, e: # if POS not found, ignore
               pass
         similarity = max(simslist)
      elif method == "WordNet_lin": # 0-1
         simslist = []
         for (s1, s2) in synsetdotproduct:
            try:
               simslist.append(s1.lin_similarity(s2, brown_ic))
            except nltk.corpus.reader.wordnet.WordNetError, e: # if POS not found, ignore
               pass
         similarity = max(simslist)
      elif method == "WordNet_jcn": # ???
         simslist = []
         for (s1, s2) in synsetdotproduct:
            try:
               simslist.append(s1.jcn_similarity(s2, brown_ic))
            except nltk.corpus.reader.wordnet.WordNetError, e: # if POS not found, ignore
               pass
         similarity = max(simslist)
      else:
         print "Sorry, I don't know this similarity (yet!)!!"
   except ValueError, e: 
      similarity = 0

   return similarity



def compareWordNet(word1, word2, method, useHeads = True):
   """
   Calculate the similarity between two words/phrases using WordNet.
   'useHead' influences the treatment of phrases.
   If set to 'true', the synsets of the head word of the phrase is used
      as representation of the phrase 
      (the WordNet standard treatment of phrases).
   If set to 'false', the synsets of all words of the phrase are used 
      as representation of the phrase 
      (the distributional semantics standard treatment of phrases).
   """

   # Decide which synsets to use
   if useHeads: # variety 1 (WordNet standard): use heads of phrases
      head1 = getHead(word1)
      head2 = getHead(word2)
      ss1 = wn.synsets(head1)
      ss2 = wn.synsets(head2)
   else: # variety 2 (distributional semantics standard): use all tokens
      ss1 = getAllSynsets(word1)
      ss2 = getAllSynsets(word2)

   # Similarity = 0 if one/both word not found
   if ss1 == [] or ss2 == []:
      return 0 # minimal similarity = 0
   
   # Get all pairs of synsets for the two words which are for the same part of speech
   # (because some of the similiarities only work in that case)
   synsetdotproduct = [(s1, s2) for (s1, s2) in product(ss1, ss2) if s1._pos == s2._pos]

   # Similarity = 0 if there are no pairs with the same POS
   if synsetdotproduct == []:
      return 0 # minimal similarity = 0

   # Get similarity as calculated by specified method
   similarity = getWNSimilarity (synsetdotproduct, method)

   # Catch value_error
   if similarity == None:
      return 0 # minimal similarity = 0
   
   return similarity



def get_distance (vectors, labels, method, useHeads = True):
   """
   Main method to select a distance measure.
   Select the measure with the variable 'method'.
   Depending on the measure, you will need to give either
      'vectors' (a numpy vector for each point) or
      'labels' (the actual word or phrase).
   Returns a distance matrix in the format that the clustering expects.
   
   Available methods:
   
   * levenshtein (labels): 
         Levenshtein / edit distance of words/phrases
   * levenshtein2 (labels): 
         Levenshtein distance of words/phrases, 
         but value is set to 0 if one word is contained in the other.
   
   * euclidean (vectors):
         Euclidean vector distance.
   
   * cosine (vectors):
         Cosine vector distance (1 - cosine similarity).
   
   * WordNet_path (labels):
   * WordNet_wup (labels):
   * WordNet_lch (labels):
   * WordNet_res (labels):
   * WordNet_lin (labels):
   * WordNet_jcn (labels):
   """


   # Levenshtein / edit distance of words/phrases
   if method == "levenshtein":
      distancematrix = []
      numb = range(len(labels))
      for i in numb:
         for j in range(i+1, len(numb)):
            distancematrix.append(distance.levenshtein(labels[i],labels[j]))
      return distancematrix


   # Levenshtein / edit distance of words/phrases
   # but value is set to 0 if one word is contained in the other, 
   # e.g., "coffee" and "coffemaker" or 
   # "coffee" and "coffee maker"
   if method == "levenshtein2":
      distancematrix = []
      numb = range(len(labels))
      for i in numb:
              for j in range(i+1, len(numb)):
                  if (labels[i] in labels[j]) or (labels[j] in labels[i]):
                     distancematrix.append(0)
                  else:
                     distancematrix.append(distance.levenshtein(labels[i],labels[j]))
      return distancematrix


   # Euclidean distance between vectors
   # Return 'None' here, because if we do not provide a matrix, the clustering
   # algorithm will calculate it itself and some linkage methods can only be used
   # if we do not provide a matrix.
   if method == "euclidean":
      return None


   # Cosine distance between vectors (1 - cosine similarity)
   if method == "cosine":
      distancematrix = spatial.distance.pdist(vectors,'cosine')
      numpy.clip(distancematrix,0,1000,distancematrix) # exclue some weird values below zero (?!)
      return distancematrix


   # WordNet distances (see 'compareWordNet' above)
   if method[0:7]== "WordNet":

      # Calculate WordNet similarities
      similaritymatrix = []
      numb = range(len(labels))
      for i in numb:
         for j in range(i+1, len(numb)):
            v = compareWordNet(labels[i], labels[j], method, useHeads)
            similaritymatrix.append(v)

      # Get the maximum similarity value
      maxvalue = max(similaritymatrix)
      
      # Convert to distances:   dist(x,y) = maxvalue - sim(x,y)
      distancematrix = []
      for value in similaritymatrix:
         distancematrix.append((maxvalue - value) / maxvalue)

      return distancematrix


   # Unknown method
   else:
      print "Sorry, I don't know this method (yet!)!!"

