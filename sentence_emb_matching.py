import argparse

parser = argparse.ArgumentParser(description='run matching using sentence embeddings and cosine similarity')
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

def load_embeddings(path, dimension = 300):
    """
    Loads the embeddings from a word2vec formatted file.
    The first line may or may not include the word count and dimension
    """
    vectors = {}
    with open(path, mode='r', encoding='utf8') as fp:
        first_line = fp.readline().rstrip('\n')
        if first_line.count(' ') == 1: # includes the "word_count dimension" information
            (word_count, dimension) = map(int, first_line.split())
        else: # assume the file only contains vectors
            fp.seek(0)
        for line in fp:
            elems = line.split()
            vectors[" ".join(elems[:-dimension])] = " ".join(elems[-dimension:])
    return vectors

source_vectors_filename = args.source_vector
target_vectors_filename = args.target_vector
vectors_source = load_embeddings(source_vectors_filename)
vectors_target = load_embeddings(target_vectors_filename)

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
    '''
    clean_corpus, clean_vectors, keys = [], {}, []
    words_we_want = set(embeddings_dictionary)
    tokenize = MosesTokenizer(language)
    for key, doc in enumerate(corpus):
        clean_doc = []
        words = tokenize(doc)
        for word in words:
            if word in words_we_want:
                clean_doc.append(word)
                clean_vectors[word] = np.array(vectors[word].split()).astype(np.float)
        if len(clean_doc) > 3 and len(clean_doc) < 25:
            keys.append(key)
        clean_corpus.append(' '.join(clean_doc))
    tokenize.close()
    return np.array(clean_corpus), clean_vectors, keys

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

vocab_counter = CountVectorizer().fit(clean_src_corpus + clean_target_corpus)
common = [w for w in vocab_counter.get_feature_names() if w in clean_src_vectors or w in clean_target_vectors]
W_common = []

for w in common:
    if w in clean_src_vectors:
        W_common.append(np.array(clean_src_vectors[w]))
    else:
        W_common.append(np.array(clean_target_vectors[w]))

print(f'{source_lang} - {target_lang}: the vocabulary size is {len(W_common)}')

from sklearn.preprocessing import normalize
W_common = np.array(W_common)
W_common = normalize(W_common) # default is l2

vect_tfidf = TfidfVectorizer(vocabulary=common, dtype=np.double, norm='l2')
vect_tfidf.fit(clean_src_corpus + clean_target_corpus)
X_idf_source = vect_tfidf.transform(clean_src_corpus)
X_idf_target = vect_tfidf.transform(clean_target_corpus)

print(f'Matrices are {X_idf_source.shape} and {W_common.shape}')
print(f'The dimensions are {X_idf_source.ndim} and {W_common.ndim}')

X_idf_source_array = X_idf_source.toarray()
X_idf_target_array = X_idf_target.toarray()
S_emb_source = np.matmul(X_idf_source_array, W_common)
S_emb_target = np.matmul(X_idf_target_array, W_common)

S_emb_target_transpose = np.transpose(S_emb_target)

cost_matrix = np.matmul(S_emb_source, S_emb_target_transpose)

from lapjv import lapjv
cost_matrix = cost_matrix * -1000
row_ind, col_ind, a = lapjv(cost_matrix, verbose=False)

result = zip(row_ind, col_ind)
hit_one = len([x for x,y in result if x == y])
print(f'{hit_one} definitions have been mapped correctly, shape of cost matrix: {str(cost_matrix.shape)}')

import csv
percentage = hit_one / instances * 100
fields = [f'{source_lang}', f'{target_lang}', f'{instances}', f'{hit_one}', f'{percentage}']

with open('semb_matcing.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)

