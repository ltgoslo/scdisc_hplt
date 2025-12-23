import gzip
import torch

input_file = "/cluster/work/projects/nn9851k/mariiaf/diachronic/eng_Latn/2011_2015_t5/1pt.gz"
with gzip.GzipFile(input_file, 'rb') as f:
    data = torch.load(f)
    for token_id, values in data.items():
        print(token_id, flush=True)
        print(len(values[0]), flush=True)
        print(len(values[1]), flush=True)
        print(values[1][1], flush=True)
        print(values[0][1], flush=True)
        break