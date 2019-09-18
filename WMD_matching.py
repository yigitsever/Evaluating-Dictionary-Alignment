###########################
#  Wasserstein Retrieval  #
###########################
import argparse

parser = argparse.ArgumentParser(description='run matching using wmd and wasserstein distances')
parser.add_argument('source_lang', help='source language short name')
parser.add_argument('target_lang', help='target language short name')
parser.add_argument('source_vector', help='path of the source vector')
parser.add_argument('target_vector', help='path of the target vector')
parser.add_argument('source_defs', help='path of the source definitions')
parser.add_argument('target_defs', help='path of the target definitions')
parser.add_argument('-n', '--instances', help='number of instances in each language to retrieve', default=2000, type=int)

args = parser.parse_args()

source_lang = args.source_lang
target_lang = args.target_lang

def load_embeddings(path, dimension=300):
    """
    Loads the embeddings from a word2vec formatted file.
    word2vec format is one line per word and it's associated embedding
    (dimension x floating numbers) separated by spaces
    The first line may or may not include the word count and dimension
    """
    vectors = {}
    with open(path, mode='r', encoding='utf8') as fp:
        first_line = fp.readline().rstrip('\n')
        if first_line.count(' ') == 1:
            # includes the "word_count dimension" information
            (word_count, dimension) = map(int, first_line.split())
        else:
            # assume the file only contains vectors
            fp.seek(0)
        for line in fp:
            elems = line.split()
            vectors[" ".join(elems[:-dimension])] = " ".join(elems[-dimension:])
    return vectors

#######################################################################
#                          Vectors Load Here                          #
#######################################################################

source_vectors_filename = args.source_vector
target_vectors_filename = args.target_vector
vectors_source = load_embeddings(source_vectors_filename)
vectors_target = load_embeddings(target_vectors_filename)

#######################################################################
#                          Corpora Load Here                          #
#######################################################################

source_defs_filename = args.source_defs
target_defs_filename = args.target_defs
defs_source = [line.rstrip('\n') for line in open(source_defs_filename, encoding='utf8')]
defs_target = [line.rstrip('\n') for line in open(target_defs_filename, encoding='utf8')]

import numpy as np
from mosestokenizer import *

def clean_corpus_using_embeddings_vocabulary(
        embeddings_dictionary,
        corpus,
        vectors,
        language,
        ):
    '''
    Cleans corpus using the dictionary of embeddings.
    Any word without an associated embedding in the dictionary is ignored.
    Adds '__target-language' and '__source-language' at the end of the words according to their language.
    '''
    clean_corpus, clean_vectors, keys = [], {}, []
    words_we_want = set(embeddings_dictionary)
    tokenize = MosesTokenizer(language)
    for key, doc in enumerate(corpus):
        clean_doc = []
        words = tokenize(doc)
        for word in words:
            if word in words_we_want:
                clean_doc.append(word + '__%s' % language)
                clean_vectors[word + '__%s' % language] = np.array(vectors[word].split()).astype(np.float)
        if len(clean_doc) > 3 and len(clean_doc) < 25:
            keys.append(key)
        clean_corpus.append(' '.join(clean_doc))
    tokenize.close()
    return np.array(clean_corpus), clean_vectors, keys

import nltk
clean_src_corpus, clean_src_vectors, src_keys = clean_corpus_using_embeddings_vocabulary(
        set(vectors_source.keys()),
        defs_source,
        vectors_source,
        source_lang,
        )

clean_target_corpus, clean_target_vectors, target_keys = clean_corpus_using_embeddings_vocabulary(
        set(vectors_target.keys()),
        defs_target,
        vectors_target,
        target_lang,
        )

import random
take = args.instances

common_keys = set(src_keys).intersection(set(target_keys))
take = min(len(common_keys), take) # you can't sample more than length
experiment_keys = random.sample(common_keys, take)

instances = len(experiment_keys)

clean_src_corpus = list(clean_src_corpus[experiment_keys])
clean_target_corpus = list(clean_target_corpus[experiment_keys])

print(f'{source_lang} - {target_lang} : document sizes: {len(clean_src_corpus)}, {len(clean_target_corpus)}')

del vectors_source, vectors_target, defs_source, defs_target

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vec = CountVectorizer().fit(clean_src_corpus + clean_target_corpus)
common = [word for word in vec.get_feature_names() if word in clean_src_vectors or word in clean_target_vectors]
W_common = []
for w in common:
    if w in clean_src_vectors:
        W_common.append(np.array(clean_src_vectors[w]))
    else:
        W_common.append(np.array(clean_target_vectors[w]))

print(f'{source_lang} - {target_lang}: the vocabulary size is {len(W_common)}')

from sklearn.preprocessing import normalize
W_common = np.array(W_common)
W_common = normalize(W_common)
vect = TfidfVectorizer(vocabulary=common, dtype=np.double, norm=None)
vect.fit(clean_src_corpus + clean_target_corpus)
X_train_idf = vect.transform(clean_src_corpus)
X_test_idf = vect.transform(clean_target_corpus)

vect_tf = CountVectorizer(vocabulary=common, dtype=np.double)
vect_tf.fit(clean_src_corpus + clean_target_corpus)
X_train_tf = vect_tf.transform(clean_src_corpus)
X_test_tf = vect_tf.transform(clean_target_corpus)

import ot
from lapjv import lapjv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import euclidean_distances
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_array
from sklearn.metrics.scorer import check_scoring
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.metrics import euclidean_distances

class WassersteinDistances(KNeighborsClassifier):
    """
    Implements a nearest neighbors classifier for input distributions using the Wasserstein distance as metric.
    Source and target distributions are l_1 normalized before computing the Wasserstein distance.
    Wasserstein is parametrized by the distances between the individual points of the distributions.
    In this work, we propose to use cross-lingual embeddings for calculating these distances.

    """
    def __init__(self, W_embed, n_neighbors=1, n_jobs=1, verbose=False, sinkhorn= False, sinkhorn_reg=0.1):
        """
        Initialization of the class.
        Arguments
        ---------
        W_embed: embeddings of the words, np.array
        verbose: True/False
        """
        self.sinkhorn = sinkhorn
        self.sinkhorn_reg = sinkhorn_reg
        self.W_embed = W_embed
        self.verbose = verbose
        super(WassersteinDistances, self).__init__(n_neighbors=n_neighbors, n_jobs=n_jobs, metric='precomputed', algorithm='brute')

    def _wmd(self, i, row, X_train):
        union_idx = np.union1d(X_train[i].indices, row.indices)
        W_minimal = self.W_embed[union_idx]
        W_dist = euclidean_distances(W_minimal)
        bow_i = X_train[i, union_idx].A.ravel()
        bow_j = row[:, union_idx].A.ravel()
        if self.sinkhorn:
            return  ot.sinkhorn2(bow_i, bow_j, W_dist, self.sinkhorn_reg, numItermax=50, method='sinkhorn_stabilized',)[0]
        else:
            return  ot.emd2(bow_i, bow_j, W_dist)

    def _wmd_row(self, row):
        X_train = self._fit_X
        n_samples_train = X_train.shape[0]
        return [self._wmd(i, row, X_train) for i in range(n_samples_train)]

    def _pairwise_wmd(self, X_test, X_train=None):
        n_samples_test = X_test.shape[0]

        if X_train is None:
            X_train = self._fit_X
        pool = Pool(nodes=self.n_jobs) # Parallelization of the calculation of the distances
        dist  = pool.map(self._wmd_row, X_test)
        return np.array(dist)

    def fit(self, X, y): # X_train_idf
        X = check_array(X, accept_sparse='csr', copy=True) # check if array is sparse
        X = normalize(X, norm='l1', copy=False)
        return super(WassersteinDistances, self).fit(X, y) # X_train_idf, np_ones(document collection size)

    def predict(self, X):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        dist = dist * 1000 # for lapjv, small floating point numbers are evil
        return super(WassersteinDistances, self).predict(dist)

    def kneighbors(self, X, n_neighbors=1): # X : X_train_idf
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        dist = dist * 1000 # for lapjv, small floating point numbers are evil
        return lapjv(dist) # and here is the matching part

def mrr_precision_at_k(golden, preds, k_list=[1,]):
    """
    Calculates Mean Reciprocal Error and Hits@1 == Precision@1
    """
    my_score = 0
    precision_at = np.zeros(len(k_list))
    for key, elem in enumerate(golden):
        if elem in preds[key]:
            location = np.where(preds[key]==elem)[0][0]
            my_score += 1/(1+ location)
        for k_index, k_value in enumerate(k_list):
            if location < k_value:
                precision_at[k_index] += 1
    return my_score/len(golden), (precision_at/len(golden))[0]

print(f'WMD - tfidf: {source_lang} - {target_lang}')
clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=14)
clf.fit(X_train_idf[:instances], np.ones(instances))
row_ind, col_ind, a = clf.kneighbors(X_test_idf[:instances], n_neighbors=instances)
result = zip(row_ind, col_ind)
hit_one = len([x for x,y in result if x == y])
print(f'{hit_one} definitions have been mapped correctly')

import csv
percentage = hit_one / instances * 100
fields = [f'{source_lang}', f'{target_lang}', f'{instances}', f'{hit_one}', f'{percentage}']
with open('/home/syigit/multilang_results/wmd_matching_result.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)

print(f'Sinkhorn - tfidf: {source_lang} - {target_lang}')
clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=14, sinkhorn=True)
clf.fit(X_train_idf[:instances], np.ones(instances))
row_ind, col_ind, a = clf.kneighbors(X_test_idf[:instances], n_neighbors=instances)

result = zip(row_ind, col_ind)
hit_one = len([x for x,y in result if x == y])
print(f'{hit_one} definitions have been mapped correctly')

percentage = hit_one / instances * 100
fields = [f'{source_lang}', f'{target_lang}', f'{instances}', f'{hit_one}', f'{percentage}']
with open('/home/syigit/multilang_results/sinkhorn_matching_result.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)