import argparse
import csv
import random

import numpy as np
from lapjv import lapjv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from Wasserstein_Distance import load_embeddings, process_corpus


def main(args):

    run_paradigm = list()

    if args.paradigm == "all":
        run_paradigm.extend(("matching", "retrieval"))
    else:
        run_paradigm.append(args.paradigm)

    source_lang = args.source_lang
    target_lang = args.target_lang
    batch = args.batch

    source_vectors_filename = args.source_vector
    target_vectors_filename = args.target_vector

    vectors_source = load_embeddings(source_vectors_filename)
    vectors_target = load_embeddings(target_vectors_filename)

    source_defs_filename = args.source_defs
    target_defs_filename = args.target_defs
    defs_source = [
        line.rstrip("\n") for line in open(source_defs_filename, encoding="utf8")
    ]
    defs_target = [
        line.rstrip("\n") for line in open(target_defs_filename, encoding="utf8")
    ]

    clean_source_corpus, clean_source_vectors, source_keys = process_corpus(
        set(vectors_source.keys()), defs_source, vectors_source, source_lang
    )

    clean_target_corpus, clean_target_vectors, target_keys = process_corpus(
        set(vectors_target.keys()), defs_target, vectors_target, target_lang
    )

    take = args.instances
    common_keys = set(source_keys).intersection(set(target_keys))
    take = min(len(common_keys), take)  # you can't sample more than length
    experiment_keys = random.sample(common_keys, take)

    instances = len(experiment_keys)

    clean_source_corpus = list(clean_source_corpus[experiment_keys])
    clean_target_corpus = list(clean_target_corpus[experiment_keys])

    if not batch:
        print(
            f"{source_lang} - {target_lang} "
            + f" document sizes: {len(clean_source_corpus)}, {len(clean_target_corpus)}"
        )

    del vectors_source, vectors_target, defs_source, defs_target

    vocab_counter = CountVectorizer().fit(clean_source_corpus + clean_target_corpus)
    common = [
        w
        for w in vocab_counter.get_feature_names()
        if w in clean_source_vectors or w in clean_target_vectors
    ]
    W_common = []

    for w in common:
        if w in clean_source_vectors:
            W_common.append(np.array(clean_source_vectors[w]))
        else:
            W_common.append(np.array(clean_target_vectors[w]))

    W_common = np.array(W_common)
    W_common = normalize(W_common)  # default is l2

    vect_tfidf = TfidfVectorizer(vocabulary=common, dtype=np.double, norm="l2")
    vect_tfidf.fit(clean_source_corpus + clean_target_corpus)
    X_idf_source = vect_tfidf.transform(clean_source_corpus)
    X_idf_target = vect_tfidf.transform(clean_target_corpus)

    X_idf_source_array = X_idf_source.toarray()
    X_idf_target_array = X_idf_target.toarray()
    S_emb_source = np.matmul(X_idf_source_array, W_common)
    S_emb_target = np.matmul(X_idf_target_array, W_common)

    S_emb_target_transpose = np.transpose(S_emb_target)

    cost_matrix = np.matmul(S_emb_source, S_emb_target_transpose)

    for paradigm in run_paradigm:
        if paradigm == "matching":

            matching_cost_matrix = cost_matrix * -1000
            row_ind, col_ind, a = lapjv(matching_cost_matrix, verbose=False)

            result = zip(row_ind, col_ind)
            hit_at_one = len([x for x, y in result if x == y])
            p_at_one = hit_at_one / instances
            percentage = hit_at_one / instances * 100

            if not batch:
                print(f"{paradigm} - semb on {source_lang} - {target_lang}")
                print(f"P @ 1: {p_at_one}")
                print(f"{percentage}% {instances} definitions")

            if batch:
                fields = [
                    f"{source_lang}",
                    f"{target_lang}",
                    f"{instances}",
                    f"{hit_at_one}",
                    f"{percentage}",
                ]

                with open("semb_matcing_results.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)

        if paradigm == "retrieval":

            hit_at_one = len(
                [x for x, y in enumerate(cost_matrix.argmax(axis=1)) if x == y]
            )
            percentage = hit_at_one / instances * 100

            if not batch:
                print(f"{hit_at_one} definitions have retrieved correctly")

            if batch:
                fields = [
                    f"{source_lang}",
                    f"{target_lang}",
                    f"{instances}",
                    f"{hit_at_one}",
                    f"{percentage}",
                ]

                with open("semb_retrieval_results.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="align dictionaries using sentence embedding representation"
    )
    parser.add_argument("source_lang", help="source language short name")
    parser.add_argument("target_lang", help="target language short name")
    parser.add_argument("source_vector", help="path of the source vector")
    parser.add_argument("target_vector", help="path of the target vector")
    parser.add_argument("source_defs", help="path of the source definitions")
    parser.add_argument("target_defs", help="path of the target definitions")
    parser.add_argument(
        "-n",
        "--instances",
        help="number of instances in each language to use",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="running in batch (store results in csv) or "
        + "running a single instance (output the results)",
    )
    parser.add_argument(
        "paradigm",
        choices=["all", "retrieval", "matching"],
        default="all",
        help="which paradigms to align with",
    )

    args = parser.parse_args()
    main(args)
