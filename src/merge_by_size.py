import argparse
from collections import defaultdict
from glob import glob
import logging
import os
import gzip
import json

import torch


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('--lang', default="cmn_Hans")
parser.add_argument('--period', default="2011_2015")
args = parser.parse_args()

in_path = f"/cluster/work/projects/nn9851k/mariiaf/diachronic/{args.lang}/{args.period}__embeddings"
out_path = f"/cluster/work/projects/nn9851k/mariiaf/diachronic/{args.lang}/{args.period}_t5__embeddings"
os.makedirs(out_path, exist_ok=True)
current_size = 0
current_data = {}
count = 0
metadata = defaultdict(list)
for file in glob(os.path.join(in_path, "*.pt.gz")):
    current_size += os.path.getsize(file)
    with gzip.GzipFile(file, "rb") as f:
        new_dict = torch.load(f)
        logger.info(f"Number of lemmas in {file}: {len(new_dict)}")
    for word in new_dict:
        embeddings = new_dict[word][0][0]
        segments = new_dict[word][1]
        first_row = embeddings[0]
        assert not first_row.any()
        embeddings = embeddings[1:, :]
        try:
            assert len(embeddings) == len(segments)
        except AssertionError:
            print(len(embeddings))
            print(len(segments))
            print(embeddings)
            print(segments)
            raise AssertionError
        if word in current_data:
            current_data[word][0] = torch.cat((current_data[word][0], embeddings), 0)
            current_data[word][1] += segments
        else:
            current_data[word] = [embeddings, segments]
        assert len(current_data[word][0]) == len(current_data[word][1])
        metadata[word[0]].append(f"{args.lang}_{count}.pt.gz")
    if current_size > 2e+9:
        with gzip.GzipFile(os.path.join(out_path, f"{args.lang}_{count}.pt.gz"), "wb") as f:
            torch.save(current_data, f)
        current_size = 0
        current_data = {}
        count += 1
with open(os.path.join(out_path, "metadata.json"), "w") as meta_f:
    json.dump(metadata, f, ensure_ascii=False)