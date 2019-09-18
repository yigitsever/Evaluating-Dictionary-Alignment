import argparse

parser = argparse.ArgumentParser(description='Run Retrieval using Sentence Embedding + Cosine')
parser.add_argument('source_lang', help='source language short name')
parser.add_argument('target_lang', help='target language short name')
parser.add_argument('source_vector', help='path of the source vector')
parser.add_argument('target_vector', help='path of the target vector')
parser.add_argument('source_defs', help='path of the source definitions')
parser.add_argument('target_defs', help='path of the target definitions')
parser.add_argument('-n', '--instances', help='number of instances in each language to retrieve', default=1000, type=int)
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

lang_source = args.source_lang
lang_target = args.target_lang

vectors_filename_source = args.source_vector
vectors_filename_target = args.target_vector

vectors_source = load_embeddings(vectors_filename_source)
vectors_target = load_embeddings(vectors_filename_target)

defs_filename_source = args.source_defs
defs_filename_target = args.target_defs
defs_source = [line.rstrip('\n') for line in open(defs_filename_source, encoding='utf8')]
defs_target = [line.rstrip('\n') for line in open(defs_filename_target, encoding='utf8')]

print('Read {} {} documents and {} {} documents'.format(len(defs_source), lang_source, len(defs_target), lang_target))

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

clean_corpus_source, clean_vectors_source, keys_source = clean_corpus_using_embeddings_vocabulary(
        set(vectors_source.keys()),
        defs_source,
        vectors_source,
        lang_source,
        )

clean_corpus_target, clean_vectors_target, keys_target = clean_corpus_using_embeddings_vocabulary(
        set(vectors_target.keys()),
        defs_target,
        vectors_target,
        lang_target,
        )

import random
take = args.instances

common_keys = set(keys_source).intersection(set(keys_target)) # definitions that fit the above requirements
take = min(len(common_keys), take) # you can't sample more than length
experiment_keys = random.sample(common_keys, take)

instances = len(experiment_keys)

clean_corpus_source = list(clean_corpus_source[experiment_keys])
clean_corpus_target = list(clean_corpus_target[experiment_keys])
print(f'{source_lang} - {target_lang} : document sizes: {len(clean_corpus_source)}, {len(clean_corpus_target)}')

del vectors_source, vectors_target, defs_source, defs_target

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vocab_counter = CountVectorizer().fit(clean_corpus_source + clean_corpus_target)
common = [w for w in vocab_counter.get_feature_names() if w in clean_vectors_source or w in clean_vectors_target]

W_common = []
for w in common:
    if w in clean_vectors_source:
        W_common.append(np.array(clean_vectors_source[w]))
    else:
        W_common.append(np.array(clean_vectors_target[w]))

print('The vocabulary size is %d' % (len(W_common)))

from sklearn.preprocessing import normalize
W_common = np.array(W_common)
W_common = normalize(W_common) # default is l2

vect_tfidf = TfidfVectorizer(vocabulary=common, dtype=np.double, norm='l2')
vect_tfidf.fit(clean_corpus_source + clean_corpus_target)
X_idf_source = vect_tfidf.transform(clean_corpus_source)
X_idf_target = vect_tfidf.transform(clean_corpus_target)

print(f'Matrices are {X_idf_source.shape} and {W_common.shape}')
print(f'The dimensions are {X_idf_source.ndim} and {W_common.ndim}')

X_idf_source_array = X_idf_source.toarray()
X_idf_target_array = X_idf_target.toarray()
S_emb_source = np.matmul(X_idf_source_array, W_common)
S_emb_target = np.matmul(X_idf_target_array, W_common)

S_emb_target_transpose = np.transpose(S_emb_target)

cost_matrix = np.matmul(S_emb_source, S_emb_target_transpose)

hit_at_one = len([x for x,y in enumerate(cost_matrix.argmax(axis=1)) if x == y])

import csv
percentage = hit_at_one / instances * 100
fields = [f'{source_lang}', f'{target_lang}', f'{instances}', f'{hit_at_one}', f'{percentage}']
with open('/home/syigit/multilang_results/sentence_emb_retrieval_axis_1.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
