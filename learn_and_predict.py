import argparse
import csv

import keras
import keras.backend as K
import numpy as np
from Helpers import Data, Get_Embedding
from keras.layers import LSTM, Embedding, Input, Lambda, concatenate
from keras.models import Model
from lapjv import lapjv
from sklearn.model_selection import train_test_split


def get_learning_rate(epoch=None, model=None):
    return np.round(float(K.get_value(model.optimizer.lr)), 5)


def make_cosine_func(hidden_size=50):
    def exponent_neg_cosine_similarity(x):
        """ Helper function for the similarity estimate of the LSTMs outputs """
        leftNorm = K.l2_normalize(x[:, :hidden_size], axis=-1)
        rightNorm = K.l2_normalize(x[:, hidden_size:], axis=-1)
        return K.sum(K.prod([leftNorm, rightNorm], axis=0), axis=1, keepdims=True)

    return exponent_neg_cosine_similarity


def main(args):

    source_lang = args.source_lang
    target_lang = args.target_lang
    hidden_size = args.hidden_size
    max_len = args.max_len
    num_iters = args.num_iters
    data_file = args.data_file
    learning_rate = args.learning_rate
    batch = args.batch

    data = Data(source_lang, target_lang, data_file, max_len)

    x_train = data.x_train
    y_train = data.y_train
    x_test = data.x_val
    y_test = data.y_val
    vocab_size = data.vocab_size
    max_len = data.max_len

    # https://stackoverflow.com/a/10741692/3005749
    x = data.y_val
    y = np.bincount(x.astype(np.int32))
    ii = np.nonzero(y)[0]
    assert ii == 1
    assert y[ii] == 1000  # hardcoded, sorry

    # separating the train-test set and the set used for our task
    (
        source_train,
        source_validate,
        target_train,
        target_validate,
        Y_train,
        Y_validate,
    ) = train_test_split(x_train[0], x_train[1], y_train)

    if not batch:
        print(f"Source Lang: {source_lang}")
        print(f"Target Lang: {target_lang}")
        print(f"Using {len(x_train[0])} pairs to learn")
        print(f"Predicting {len(Y_validate)} pairs")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Maximum sequence length: {max_len}")

    source_emb_file = args.source_emb_file
    target_emb_file = args.target_emb_file

    embedding = Get_Embedding(
        source_lang, target_lang, source_emb_file, target_emb_file, data.word_to_id
    )
    embedding_size = embedding.embedding_matrix.shape[1]

    seq_1 = Input(shape=(max_len,), dtype="int32", name="sequence1")
    seq_2 = Input(shape=(max_len,), dtype="int32", name="sequence2")

    embed_layer = Embedding(
        output_dim=embedding_size,
        input_dim=vocab_size + 1,
        input_length=max_len,
        trainable=False,
    )
    embed_layer.build((None,))
    embed_layer.set_weights([embedding.embedding_matrix])

    input_1 = embed_layer(seq_1)
    input_2 = embed_layer(seq_2)

    l1 = LSTM(units=300, use_bias=True, dropout=0.05, recurrent_dropout=0.1)

    l1_out = l1(input_1)
    l2_out = l1(input_2)

    concats = concatenate([l1_out, l2_out], axis=-1)

    out_func = make_cosine_func(hidden_size)

    main_output = Lambda(out_func, output_shape=(1,))(concats)

    model = Model(inputs=[seq_1, seq_2], outputs=[main_output])

    opt = keras.optimizers.Adadelta(lr=learning_rate, clipnorm=1.2)

    model.compile(optimizer=opt, loss="mean_squared_error", metrics=["accuracy"])
    model.summary()

    adjuster = keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", patience=5, verbose=1, factor=0.5, min_lr=0.0001
    )

    model.fit(
        x=[source_train, target_train],
        y=Y_train,
        validation_data=([source_validate, target_validate], Y_validate),
        epochs=num_iters,
        batch_size=32,
        verbose=0,
        callbacks=[adjuster],
    )

    cost_matrix = None

    target_sents = x_test[1]
    precision_at_one = 0
    for index, sent in enumerate(x_test[0]):
        source_sents = np.array([sent] * 1000)
        to_predict = [source_sents, target_sents]
        preds = model.predict(to_predict)

        if index == 0:
            cost_matrix = np.c_[preds]
        else:
            cost_matrix = np.c_[cost_matrix, preds]

        if np.argmax(preds.ravel()) == index:
            precision_at_one += 1

    cost_matrix = cost_matrix * -1000

    row_ind, col_ind, a = lapjv(cost_matrix, verbose=False)

    result = zip(row_ind, col_ind)
    hit_one = len([x for x, y in result if x == y])

    matching_pa1 = hit_one / 1000 * 100

    training_samples = len(Y_train)
    validation_samples = len(Y_validate)
    test_samples = len(y_test)
    fields = [
        source_lang,
        target_lang,
        training_samples,
        validation_samples,
        test_samples,
        precision_at_one,
        matching_pa1,
    ]

    if not batch:
        print(f"Supervised Retrieval {source_lang} - {target_lang}")
        print(f"P@1: {precision_at_one/1000}")
    else:
        with open("supervised_matching.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(fields)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-sl", "--source_lang", type=str, help="Source language.", required=True
    )
    parser.add_argument(
        "-tl", "--target_lang", type=str, help="Target language.", required=True
    )
    parser.add_argument(
        "-df", "--data_file", type=str, help="Path to dataset.", required=True
    )
    parser.add_argument(
        "-es",
        "--source_emb_file",
        type=str,
        help="Path to source embedding file.",
        required=True,
    )
    parser.add_argument(
        "-et",
        "--target_emb_file",
        type=str,
        help="Path to target embedding file.",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--max_len",
        type=int,
        help="Maximum number of words in a sentence.",
        default=25,
    )
    parser.add_argument(
        "-z",
        "--hidden_size",
        type=int,
        help="Number of units in LSTM layer.",
        default=300,
    )
    parser.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="running in batch (store results to csv) or "
        + "running in a single instance (output the results)",
    )
    parser.add_argument(
        "-n", "--num_iters", type=int, help="Number of iterations/epochs.", default=30
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="Learning rate for optimizer.",
        default=0.5,
    )

    args = parser.parse_args()
    main(args)
