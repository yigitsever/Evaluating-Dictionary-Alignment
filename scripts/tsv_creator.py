import argparse
import csv
import math
import random
import unicodedata
from re import sub

from mosestokenizer import MosesTokenizer


def load_def_from_file(def_filename):
    """
    def_filename: full path of the definition file
    returns: list of dictionary definitions

    """
    return [
        unicodedata.normalize("NFC", line.rstrip("\n"))
        for line in open(def_filename, encoding="utf8")
    ]


def clean_corpus_suffix(corpus, language):
    """
    Adds '__target-language' and '__source-language' at the end of the words
    """
    clean_corpus = []
    tokenize = MosesTokenizer(language)
    for definition in corpus:
        definition = sub(r"'", "", definition)
        definition = sub(r"[^\w]", " ", definition)
        clean_doc = []
        words = tokenize(definition)
        for word in words:
            clean_doc.append(word + "__%s" % language)
        clean_corpus.append(" ".join(clean_doc))
    return clean_corpus


def create_pos_neg_samples(length):
    indices = list(range(length))
    halfsize = math.ceil(length / 2)
    neg_points = random.sample(indices, halfsize)
    neg_indices = list(neg_points)
    random.shuffle(neg_indices)

    for (index, point) in zip(neg_indices, neg_points):
        indices[point] = index
    labels = [1] * length
    for i in neg_points:
        labels[i] = 0

    return indices, labels


def main(args):

    source_lang = args.source_lang
    target_lang = args.target_lang

    source_defs_filename = args.source_defs
    target_defs_filename = args.target_defs
    defs_source = load_def_from_file(source_defs_filename)
    defs_target = load_def_from_file(target_defs_filename)

    clean_source_corpus = clean_corpus_suffix(defs_source, source_lang)
    clean_target_corpus = clean_corpus_suffix(defs_target, target_lang)

    assert len(clean_source_corpus) == len(clean_target_corpus)

    set_aside = args.set_aside

    source_predict = clean_source_corpus[-set_aside:]
    target_predict = clean_target_corpus[-set_aside:]
    labels_predict = [1] * set_aside

    # placeholder, won't be used, we can use 1 because they're correct

    clean_source_corpus = clean_source_corpus[:-set_aside]
    clean_target_corpus = clean_target_corpus[:-set_aside]

    size = len(clean_source_corpus)

    while True:
        indices, labels = create_pos_neg_samples(size)
        shuffled_target = [clean_target_corpus[index] for index in indices]
        check = [
            clean
            for clean, shuf in zip(clean_target_corpus, shuffled_target)
            if clean == shuf
        ]
        halfsize = math.ceil(size / 2)
        try:
            assert len(check) == halfsize
        except AssertionError:
            pass
        else:
            break

    assert len(clean_source_corpus) == len(shuffled_target) == size
    assert len(labels) == len(clean_source_corpus) == len(shuffled_target)

    with open(
        f"{source_lang}_to_{target_lang}.tsv", "w", encoding="utf8", newline=""
    ) as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter="\t", lineterminator="\n")
        tsv_writer.writerow(
            [f"{source_lang} definition", f"{target_lang} definition", "is same"]
        )
        for row in zip(clean_source_corpus, shuffled_target, labels):
            tsv_writer.writerow(row)
        for row in zip(source_predict, target_predict, labels_predict):
            tsv_writer.writerow(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create .tsv file from two wordnet definitions"
    )
    parser.add_argument("source_lang", help="source language short name")
    parser.add_argument("target_lang", help="target language short name")
    parser.add_argument("source_defs", help="path of the source definitions")
    parser.add_argument("target_defs", help="path of the target definitions")
    parser.add_argument(
        "-n", "--set_aside", help="set aside to validate on", type=int, default=1000
    )
    args = parser.parse_args()
    main(args)
