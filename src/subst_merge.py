import sys
import os
import logging
import json
from smart_open import open

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def merge_substitutions(main_dic, new_dic):
    for word in new_dic:
        substitutions = new_dic[word][0]  # list
        segment = new_dic[word][1]  # list (text and id)
        if word in main_dic:
            main_dic[word][0].append(substitutions)
            main_dic[word][1].append(segment)
        else:
            main_dic[word] = [[substitutions], [segment]]
        assert len(main_dic[word][0]) == len(main_dic[word][1])
    return main_dic


if __name__ == "__main__":
    input_data = sys.argv[1]
    output_data = sys.argv[2]
    os.makedirs(output_data, exist_ok=True)

    with os.scandir(input_data) as period:
        for entry in period:
            if entry.name.endswith(".jsonl.gz"):
                initial = entry.name[0]
                accum_data = {}
                initial_file = f"{input_data}/{entry.name}"
                logger.info(initial_file)
                with open(initial_file, "r") as f:
                    for line in f:
                        instance = json.loads(line.strip())
                        accum_data = merge_substitutions(accum_data, instance)
                logger.info(len(accum_data))
                with open(f"{output_data}/{initial}.json.zst", "w") as of:
                    of.write(
                        json.dumps(
                            accum_data, sort_keys=True, indent=4, ensure_ascii=False
                        )
                    )
                    logger.info(
                        f"Merged file saved as {output_data}/{initial}.json.zst"
                    )
