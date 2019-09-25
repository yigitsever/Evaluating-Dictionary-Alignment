import argparse
import csv
import random

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from Wasserstein_Distance import (WassersteinMatcher, WassersteinRetriever,
                                  load_embeddings, process_corpus)


def main(args):

    np.seterr(divide="ignore")  # POT has issues with divide by zero errors
    source_lang = args.source_lang
    target_lang = args.target_lang

    source_vectors_filename = args.source_vector
    target_vectors_filename = args.target_vector
    vectors_source = load_embeddings(source_vectors_filename)
    vectors_target = load_embeddings(target_vectors_filename)

    source_defs_filename = args.source_defs
    target_defs_filename = args.target_defs

    batch = args.batch
    input_mode = args.mode
    input_paradigm = args.paradigm

    run_method = list()
    run_paradigm = list()

    if input_paradigm == "all":
        run_paradigm.extend(("matching", "retrieval"))
    else:
        run_paradigm.append(input_paradigm)

    if input_mode == "all":
        run_method.extend(["wmd", "snk"])
    else:
        run_method.append(input_mode)

    defs_source = [
        line.rstrip("\n") for line in open(source_defs_filename, encoding="utf8")
    ]
    defs_target = [
        line.rstrip("\n") for line in open(target_defs_filename, encoding="utf8")
    ]

    clean_src_corpus, clean_src_vectors, src_keys = process_corpus(
        set(vectors_source.keys()), defs_source, vectors_source, source_lang
    )

    clean_target_corpus, clean_target_vectors, target_keys = process_corpus(
        set(vectors_target.keys()), defs_target, vectors_target, target_lang
    )

    take = args.instances

    common_keys = set(src_keys).intersection(set(target_keys))
    take = min(len(common_keys), take)  # you can't sample more than length
    experiment_keys = random.sample(common_keys, take)

    instances = len(experiment_keys)

    clean_src_corpus = list(clean_src_corpus[experiment_keys])
    clean_target_corpus = list(clean_target_corpus[experiment_keys])

    del vectors_source, vectors_target, defs_source, defs_target

    vec = CountVectorizer().fit(clean_src_corpus + clean_target_corpus)
    common = [
        word
        for word in vec.get_feature_names()
        if word in clean_src_vectors or word in clean_target_vectors
    ]
    W_common = []
    for w in common:
        if w in clean_src_vectors:
            W_common.append(np.array(clean_src_vectors[w]))
        else:
            W_common.append(np.array(clean_target_vectors[w]))

    if not batch:
        print(
            f"{source_lang} - {target_lang}\n"
            + f" document sizes: {len(clean_src_corpus)}, {len(clean_target_corpus)}\n"
            + f" vocabulary size: {len(W_common)}"
        )

    W_common = np.array(W_common)
    W_common = normalize(W_common)
    vect = TfidfVectorizer(vocabulary=common, dtype=np.double, norm=None)
    vect.fit(clean_src_corpus + clean_target_corpus)
    X_train_idf = vect.transform(clean_src_corpus)
    X_test_idf = vect.transform(clean_target_corpus)

    for paradigm in run_paradigm:
        WassersteinDriver = None
        if paradigm == "matching":
            WassersteinDriver = WassersteinMatcher
        else:
            WassersteinDriver = WassersteinRetriever

        for metric in run_method:
            if not batch:
                print(f"{paradigm} - {metric} on {source_lang} - {target_lang}")

            clf = WassersteinDriver(
                W_embed=W_common, n_neighbors=5, n_jobs=14, sinkhorn=(metric == "snk")
            )
            clf.fit(X_train_idf[:instances], np.ones(instances))
            p_at_one, percentage = clf.align(
                X_test_idf[:instances], n_neighbors=instances
            )

            if not batch:
                print(f"P @ 1: {p_at_one}\n{percentage}% {instances} definitions\n")
            else:
                fields = [
                    f"{source_lang}",
                    f"{target_lang}",
                    f"{instances}",
                    f"{p_at_one}",
                    f"{percentage}",
                ]
                with open(f"{metric}_{paradigm}_results.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="align dictionaries using wmd and wasserstein distance"
    )
    parser.add_argument("source_lang", help="source language short name")
    parser.add_argument("target_lang", help="target language short name")
    parser.add_argument("source_vector", help="path of the source vector")
    parser.add_argument("target_vector", help="path of the target vector")
    parser.add_argument("source_defs", help="path of the source definitions")
    parser.add_argument("target_defs", help="path of the target definitions")
    parser.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="running in batch (store results in csv) or"
        + "running a single instance (output the results)",
    )
    parser.add_argument(
        "mode",
        choices=["all", "wmd", "snk"],
        default="all",
        help="which methods to run",
    )
    parser.add_argument(
        "paradigm",
        choices=["all", "retrieval", "matching"],
        default="all",
        help="which paradigms to align with",
    )
    parser.add_argument(
        "-n",
        "--instances",
        help="number of instances in each language to retrieve",
        default=1000,
        type=int,
    )

    args = parser.parse_args()

    main(args)
