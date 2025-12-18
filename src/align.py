# python3
# coding: utf-8

import argparse
import logging
import gensim
import numpy as np
from os import path


def smart_procrustes_align_gensim(
    base_embed: gensim.models.KeyedVectors, other_embed: gensim.models.KeyedVectors
):
    """
    Procrustes analysis to make two word embeddings compatible.
    :param base_embed: first embedding array
    :param other_embed: second embedding array to be changed
    :return other_embed: changed embedding array
    """

    shared_vocab = list(
        set(base_embed.key_to_index).intersection(other_embed.key_to_index)
    )

    base_idx2word = {num: word for num, word in enumerate(base_embed.index_to_key)}
    other_idx2word = {num: word for num, word in enumerate(other_embed.index_to_key)}

    base_word2idx = {word: num for num, word in base_idx2word.items()}
    other_word2idx = {word: num for num, word in other_idx2word.items()}

    base_shared_indices = [
        base_word2idx[word] for word in shared_vocab
    ]  # remember to remove tqdm
    other_shared_indices = [
        other_word2idx[word] for word in shared_vocab
    ]  # remember to remove tqdm

    base_vecs = base_embed.get_normed_vectors()
    other_vecs = other_embed.get_normed_vectors()

    base_shared_vecs = base_vecs[base_shared_indices]
    other_shared_vecs = other_vecs[other_shared_indices]

    m = other_shared_vecs.T @ base_shared_vecs
    u, _, v = np.linalg.svd(m)
    ortho = u @ v

    # Replace original array with modified one
    # i.e. multiplying the embedding matrix by "ortho"
    other_embed.vectors = other_vecs.dot(ortho)

    return other_embed, shared_vocab


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--emb0", "-e0", help="Base model", required=True)
    arg("--emb1", "-e1", help="Model to align with the base one", required=True)
    arg("--changes", "-c", help="How many most changed words to show?", type=int)
    args = parser.parse_args()

    models = []

    for mfile in [args.emb0, args.emb1]:
        model = gensim.models.KeyedVectors.load(mfile)
        models.append(model)

    logger.info("Aligning models...")
    models[1], shared_vocabulary = smart_procrustes_align_gensim(models[0], models[1])
    logger.info("Alignment complete")

    directory, filename = path.split(args.emb1)
    new_name = filename.split(".")[0] + "_aligned"
    newpath = path.join(directory, new_name + ".bin")
    models[1].save_word2vec_format(newpath, binary=True)
    logger.info(f"Aligned model saved to {newpath}")

    if args.changes:
        similarities = [
            np.dot(
                models[0].get_vector(word, norm=True),
                models[1].get_vector(word, norm=True),
            )
            for word in shared_vocabulary
        ]
        nr_elements = args.changes
        min_indices = np.argpartition(similarities, nr_elements)[:nr_elements]
        shifts = [
            f"{similarities[el]:0.3}\t{shared_vocabulary[el]}" for el in min_indices
        ]
        logger.info("Most changed words (with similarities):")
        for el in sorted(shifts):
            logger.info(el)
