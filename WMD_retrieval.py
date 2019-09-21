import argparse
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from Wasserstein_Distance import Wasserstein_Retriever
from Wasserstein_Distance import load_embeddings, clean_corpus_using_embeddings_vocabulary, mrr_precision_at_k
import csv
import sys

def main(args):

    np.seterr(divide='ignore') # POT has issues with divide by zero errors
    source_lang = args.source_lang
    target_lang = args.target_lang

    source_vectors_filename = args.source_vector
    target_vectors_filename = args.target_vector
    vectors_source = load_embeddings(source_vectors_filename)
    vectors_target = load_embeddings(target_vectors_filename)

    source_defs_filename = args.source_defs
    target_defs_filename = args.target_defs

    batch = args.batch
    mode = args.mode
    runfor = list()

    if (mode == 'all'):
        runfor.extend(['wmd','snk'])
    else:
        runfor.append(mode)

    defs_source = [line.rstrip('\n') for line in open(source_defs_filename, encoding='utf8')]
    defs_target = [line.rstrip('\n') for line in open(target_defs_filename, encoding='utf8')]

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

    take = args.instances

    common_keys = set(src_keys).intersection(set(target_keys))
    take = min(len(common_keys), take) # you can't sample more than length
    experiment_keys = random.sample(common_keys, take)

    instances = len(experiment_keys)

    clean_src_corpus = list(clean_src_corpus[experiment_keys])
    clean_target_corpus = list(clean_target_corpus[experiment_keys])

    if (not batch):
        print(f'{source_lang} - {target_lang} : document sizes: {len(clean_src_corpus)}, {len(clean_target_corpus)}')

    del vectors_source, vectors_target, defs_source, defs_target

    vec = CountVectorizer().fit(clean_src_corpus + clean_target_corpus)
    common = [word for word in vec.get_feature_names() if word in clean_src_vectors or word in clean_target_vectors]
    W_common = []
    for w in common:
        if w in clean_src_vectors:
            W_common.append(np.array(clean_src_vectors[w]))
        else:
            W_common.append(np.array(clean_target_vectors[w]))

    if (not batch):
        print(f'{source_lang} - {target_lang}: the vocabulary size is {len(W_common)}')

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

    for metric in runfor:
        if (not batch):
            print(f'{metric} - tfidf: {source_lang} - {target_lang}')

        clf = Wasserstein_Retriever(W_embed=W_common, n_neighbors=5, n_jobs=14, sinkhorn=(metric == 'snk'))
        clf.fit(X_train_idf[:instances], np.ones(instances))
        dist, preds = clf.kneighbors(X_test_idf[:instances], n_neighbors=instances)
        mrr, p_at_1 = mrr_precision_at_k(list(range(len(preds))), preds)
        percentage = p_at_1 * 100

        if (not batch):
            print(f'MRR: {mrr} | Precision @ 1: {p_at_1}')
        else:
            fields = [f'{source_lang}', f'{target_lang}', f'{instances}', f'{mrr}', f'{p_at_1}', f'{percentage}']
            with open(f'{metric}_retrieval_result.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='run retrieval using wmd or snk')
    parser.add_argument('source_lang', help='source language short name')
    parser.add_argument('target_lang', help='target language short name')
    parser.add_argument('source_vector', help='path of the source vector')
    parser.add_argument('target_vector', help='path of the target vector')
    parser.add_argument('source_defs', help='path of the source definitions')
    parser.add_argument('target_defs', help='path of the target definitions')
    parser.add_argument('-b', '--batch', action='store_true', help='running in batch (store results in csv) or running a single instance (output the results)')
    parser.add_argument('mode', choices=['all', 'wmd', 'snk'], default='all', help='which methods to run')
    parser.add_argument('-n', '--instances', help='number of instances in each language to retrieve', default=1000, type=int)

    args = parser.parse_args()

    main(args)
