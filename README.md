# AspectClustering
Hierarchical clustering of aspect phrases with scipy

## Files
- `scipy_clustering.py` - the main programm that does the clustering
- `distancemeasures.py` - some distance measures between aspect phrases to chose from
- `vectorexample.txt` - an example file to play around, containing phrases and vectors (from word2vec)

## Stuff you'll need
- [SciPy](https://www.scipy.org/)
- [NLTK WordNet Interface](http://www.nltk.org/howto/wordnet.html)
  (if you don't want to install/use WordNet, just throw out everything related to it in `distancemeasures.py`)

## Usage
`python scipy_clustering.py <input file> <distance measure> <cluster linkage> <number of clusters>`

Example:
`python scipy_clustering.py vectorexample.txt euclid single 5`

See code for the possible values.

## Licence and Warranty
Do whatever you want. This is code from a student project, no guarantees given, no support.
