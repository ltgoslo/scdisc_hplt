import torch
import gzip
import sys
import os
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def merge_embeddings(main_dic, new_dic):
    for word in new_dic:
        embeddings = new_dic[word][0][0]
        segments = new_dic[word][1]
        first_row = embeddings[0]
        assert not first_row.any()
        embeddings = embeddings[1:, :]
        assert len(embeddings) == len(segments)
        if word in main_dic:
            main_dic[word][0] = torch.cat((main_dic[word][0], embeddings), 0)
            main_dic[word][1] += segments
        else:
            main_dic[word] = [embeddings, segments]
        assert len(main_dic[word][0]) == len(main_dic[word][1])
    return main_dic


initials = set()

input_data = sys.argv[1]
output_data = sys.argv[2]

with os.scandir(input_data) as period:
    for entry in period:
        if entry.name.endswith(".pt.gz"):
            initials.add(entry.name[0])

for el in initials:
    logger.info(el)
    accum_data = {}
    with os.scandir(sys.argv[1]) as period:
        for entry in period:
            if entry.name == "target_ids.pt.gz":
                continue
            if entry.name.endswith(".pt.gz") and entry.name.startswith(el):
                logger.info(entry.name)
                with gzip.GzipFile(os.path.join(sys.argv[1], entry.name), "rb") as f:
                    data = torch.load(f)
                    logger.info(len(data))
                    accum_data = merge_embeddings(accum_data, data)
    logger.info(len(accum_data))
    with gzip.open(f"{output_data}/{el}.pt.gz", "wb") as of:
        torch.save(accum_data, of)
        logger.info(f"Merged file saved as {output_data}/{el}.pt.gz")
