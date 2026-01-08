import torch
import gzip
import sys
import os
import logging
from torch_merge import merge_embeddings

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


split_by_initial = False  # If false, will save one large file per period

accum_data = {}

input_data = sys.argv[1]
output_data = sys.argv[2]


with os.scandir(sys.argv[1]) as period:
    for entry in period:
        if entry.name == "target_ids.pt.gz":
            continue
        if entry.name.endswith(".pt.gz"):
            logger.info(entry.name)
            with gzip.GzipFile(os.path.join(sys.argv[1], entry.name), "rb") as f:
                data = torch.load(f)
                logger.info(len(data))
                accum_data = merge_embeddings(accum_data, data)
    logger.info(len(accum_data))


if split_by_initial:
    logger.info("Now splitting..")
    initials = set()

    for word in accum_data.keys():
        initial = word[0]
        initials.add(initial)

    logger.info(initials)

    for initial in initials:
        data = {k: accum_data[k] for k in accum_data.keys() if k[0] == initial}
        with gzip.open(f"{output_data}/{initial}.pt.gz", "wb") as of:
            torch.save(data, of)
            logger.info(f"Merged file saved as {output_data}/{initial}.pt.gz")
else:
    with gzip.open(f"{output_data}/1.pt.gz", "wb") as of:
        torch.save(accum_data, of)
        logger.info(f"Merged file saved as {output_data}/1.pt.gz")
