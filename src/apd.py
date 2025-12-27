from collections import defaultdict
from itertools import combinations
from glob import glob
import gzip
import os

import torch
from torch.nn import CosineSimilarity
from tqdm import tqdm


periods = {}
target_words_changed = ["ai", "remote"]
target_words_not_changed = ["legislative", "jurisdiction"]
words = target_words_changed + target_words_not_changed
cos = CosineSimilarity(dim=1, eps=1e-6)
for period in ("2011_2015", "2020_2021", "2024_2025"):
    print(period, flush=True)
    embeddings_dir = f"/cluster/work/projects/nn9851k/mariiaf/diachronic/eng_Latn/{period}_t5/"
    word_embeddings = defaultdict(list)
    for word in words:
        print(word, flush=True)
        first_letter = word[0]
        for input_file in tqdm(glob(os.path.join(embeddings_dir, f"{first_letter}*.pt.gz"))):
            with gzip.GzipFile(input_file, 'rb') as f:
                data = torch.load(f)
                if data.get(word) is not None:
                    embeddings = data[word][0][0][1:, :]
                    assert embeddings.size()[1] == 768
                    word_embeddings[word].append(embeddings)
    for word, word_embedding in word_embeddings.items():
        word_embedding = torch.cat(word_embedding, 0)
        assert word_embedding.size()[1] == 768
        word_embeddings[word] = word_embedding
    periods[period] = word_embeddings
for combination in combinations(periods, 2):
    print(combination, flush=True)
    word_embeddings_dict_first = periods[combination[0]]
    word_embeddings_dict_second = periods[combination[1]]
    for word in words:
        print(word, flush=True)
        first_embedding = word_embeddings_dict_first[word]
        second_embedding = word_embeddings_dict_second[word]
        print(first_embedding.size())
        print(second_embedding.size())
        samples_1 = first_embedding.size()[0]
        samples_2 = second_embedding.size()[0]
        min_samples = min(samples_1, samples_2, 1000)
        first_embedding = first_embedding[:min_samples, :]
        second_embedding = second_embedding[:min_samples, :]
        assert first_embedding.size()[1] == 768
        assert second_embedding.size()[1] == 768
        sim = cos(first_embedding, second_embedding)
        print(sim.size(), flush=True)
        print(1 - torch.mean(sim), flush=True)
