#!/bin/env python3
# coding: utf-8

import gensim
import logging
import multiprocessing
import argparse
from os import path, makedirs

# This script trains a word2vec word embedding model using Gensim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--corpus", help="Path to a training corpus (can be compressed)", required=True)
    arg("--cores", default=16, help="Limit on the number of CPU cores to use")
    arg("--sg", default=1, type=int, help="Use Skipgram (1) or CBOW (0)")
    arg("--window", default=10, type=int, help="Size of context window")
    arg("--vocab", default=50000, type=int, help="Max vocabulary size")
    args = parser.parse_args()

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # This will be our training corpus to infer word embeddings from.
    # Most probably, a compressed text file, one doc/sentence per line:
    corpus = args.corpus

    # Iterator over lines of the corpus
    data = gensim.models.word2vec.LineSentence(corpus)

    # How many workers (CPU cores) to use during the training?
    if args.cores:
        # Use the number of cores we are told to use (in a SLURM file, for example):
        cores = int(args.cores)
    else:
        # Use all cores we have access to except one
        cores = (
            multiprocessing.cpu_count() - 1
        )
    logger.info(f"Number of cores to use: {cores}")

    # Setting up training hyperparameters:
    # Use Skipgram (1) or CBOW (0) algorithm?
    skipgram = args.sg
    # Context window size (e.g., 2 words to the right and to the left)
    window = args.window
    # How many words types we want to be considered (sorted by frequency)?
    vocabsize = args.vocab

    vectorsize = 300  # Dimensionality of the resulting word embeddings.

    # For how many epochs to train a model (how many passes over corpus)?
    iterations = 5

    # Start actual training!

    # NB: Subsampling ('sample' parameter) is used to stochastically downplay the influence
    # of very frequent words. If our corpus is already filtered for stop words
    # (functional parts of speech), we do not need subsampling and set it to zero.
    model = gensim.models.Word2Vec(
        data,
        vector_size=vectorsize,
        window=window,
        workers=cores,
        sg=skipgram,
        max_final_vocab=vocabsize,
        epochs=iterations,
        sample=0.001,
    )

    directory, filename = path.split(corpus)

    word2vec_dir = directory + "/word2vec/"

    makedirs(word2vec_dir, exist_ok=True)

    # Saving the resulting model to a file
    filename = filename.replace(".txt.zst", ".model")
    logger.info(filename)

    # Save the model without the output vectors
    model.wv.save(word2vec_dir + filename)

    # Save the model in the word2vec binary format
    model.wv.save_word2vec_format(word2vec_dir + filename.replace(".model", ".bin"), binary=True)

    # model.save(filename)  # If you intend to train the model further
