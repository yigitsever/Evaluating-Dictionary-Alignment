import itertools

import numpy as np
from sklearn.model_selection import train_test_split as split_data

import pandas as pd
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences


class Data(object):
    def __init__(
        self,
        source_lang,
        target_lang,
        data_file,
        max_len=None,
        instances=1000,
        vocab_limit=None,
        sentence_cols=None,
        score_col=None,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.data_file = data_file
        self.max_len = max_len
        self.instances = instances
        self.vocab_size = 1
        self.vocab_limit = vocab_limit

        if sentence_cols is None:
            self.sequence_cols = [
                f"{source_lang} definition",
                f"{target_lang} definition",
            ]
        else:
            self.sequence_cols = sentence_cols

        if score_col is None:
            self.score_col = "is same"
        else:
            self.score_col = score_col

        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.vocab = set("PAD")
        self.word_to_id = {"PAD": 0}
        self.id_to_word = {0: "PAD"}
        self.word_to_count = dict()
        self.run()

    def text_to_word_list(self, text):
        """ Pre process and convert texts to a list of words """
        text = str(text)
        text = text.split()
        return text

    def load_data(self):
        # Load data set
        data_df = pd.read_csv(self.data_file, sep="\t")

        # Iterate over required sequences of provided dataset
        for index, row in data_df.iterrows():
            # Iterate through the text of both questions of the row
            for sequence in self.sequence_cols:
                s2n = []  # Sequences with words replaces with indices
                for word in self.text_to_word_list(row[sequence]):
                    if word not in self.vocab:
                        self.vocab.add(word)
                        self.word_to_id[word] = self.vocab_size
                        self.word_to_count[word] = 1
                        s2n.append(self.vocab_size)
                        self.id_to_word[self.vocab_size] = word
                        self.vocab_size += 1
                    else:
                        self.word_to_count[word] += 1
                        s2n.append(self.word_to_id[word])

                # Replace |sequence as word| with |sequence as number| representation
                data_df.at[index, sequence] = s2n
        return data_df

    def pad_sequences(self):
        if self.max_len == 0:
            self.max_len = max(
                max(len(seq) for seq in self.x_train[0]),
                max(len(seq) for seq in self.x_train[1]),
                max(len(seq) for seq in self.x_val[0]),
                max(len(seq) for seq in self.x_val[1]),
            )

        # Zero padding
        for dataset, side in itertools.product([self.x_train, self.x_val], [0, 1]):
            if self.max_len:
                dataset[side] = pad_sequences(dataset[side], maxlen=self.max_len)
            else:
                dataset[side] = pad_sequences(dataset[side])

    def run(self):
        # Loading data and building vocabulary.
        data_df = self.load_data()

        X = data_df[self.sequence_cols]
        Y = data_df[self.score_col]

        self.x_train, self.x_val, self.y_train, self.y_val = split_data(
            X, Y, test_size=self.instances, shuffle=False
        )

        # Split to lists
        self.x_train = [self.x_train[column] for column in self.sequence_cols]
        self.x_val = [self.x_val[column] for column in self.sequence_cols]

        # Convert labels to their numpy representations
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values

        # Padding Sequences.
        self.pad_sequences()


class Get_Embedding(object):
    def __init__(self, source_lang, target_lang, source_emb, target_emb, word_index):
        self.embedding_size = 300  # Default dimensionality
        self.embedding_matrix = self.create_embed_matrix(
            source_lang, target_lang, source_emb, target_emb, word_index
        )

    def create_embed_matrix(
        self, source_lang, target_lang, source_emb, target_emb, word_index
    ):
        source_vecs = KeyedVectors.load_word2vec_format(source_emb)
        target_vecs = KeyedVectors.load_word2vec_format(target_emb)

        # Prepare Embedding Matrix.
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_size))

        # word has either __source or __target appended
        for key, i in word_index.items():
            if "__" not in key:
                print("Skipping {}".format(key))
                continue

            word, lang = key.split("__")

            if lang == source_lang:
                if word in source_vecs.vocab:
                    embedding_matrix[i] = source_vecs.word_vec(word)
            else:
                if word in target_vecs.vocab:
                    embedding_matrix[i] = target_vecs.word_vec(word)

        del source_vecs
        del target_vecs
        return embedding_matrix
