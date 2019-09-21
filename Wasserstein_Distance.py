import ot
from sklearn.preprocessing import normalize
from lapjv import lapjv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import euclidean_distances
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_array
from sklearn.metrics.scorer import check_scoring
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.metrics import euclidean_distances
import numpy as np
from mosestokenizer import MosesTokenizer

class Wasserstein_Matcher(KNeighborsClassifier):
    """
    Implements a nearest neighbors classifier for input distributions using the Wasserstein distance as metric.
    Source and target distributions are l_1 normalized before computing the Wasserstein distance.
    Wasserstein is parametrized by the distances between the individual points of the distributions.
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
        super(Wasserstein_Matcher, self).__init__(n_neighbors=n_neighbors, n_jobs=n_jobs, metric='precomputed', algorithm='brute')

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
        return super(Wasserstein_Matcher, self).fit(X, y) # X_train_idf, np_ones(document collection size)

    def predict(self, X):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        dist = dist * 1000 # for lapjv, small floating point numbers are evil
        return super(Wasserstein_Matcher, self).predict(dist)

    def kneighbors(self, X, n_neighbors=1): # X : X_train_idf
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        dist = dist * 1000 # for lapjv, small floating point numbers are evil
        return lapjv(dist) # and here is the matching part


class Wasserstein_Retriever(KNeighborsClassifier):
    """
    Implements a nearest neighbors classifier for input distributions using the Wasserstein distance as metric.
    Source and target distributions are l_1 normalized before computing the Wasserstein distance.
    Wasserstein is parametrized by the distances between the individual points of the distributions.
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
        super(Wasserstein_Retriever, self).__init__(n_neighbors=n_neighbors, n_jobs=n_jobs, metric='precomputed', algorithm='brute')

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

    def fit(self, X, y):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        return super(Wasserstein_Retriever, self).fit(X, y)

    def predict(self, X):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        return super(Wasserstein_Retriever, self).predict(dist)

    def kneighbors(self, X, n_neighbors=1):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        return super(Wasserstein_Retriever, self).kneighbors(dist, n_neighbors)

    def align(self, X, n_neighbors=1):
        """
        Wrapper function over kneighbors to return
        precision at one and percentage values

        """
        dist, preds = self.kneighbors(X, n_neighbors)
        mrr, p_at_one = mrr_precision_at_k(list(range(len(preds))), preds)
        percentage = p_at_one * 100
        return (p_at_one, percentage)


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

