import gzip
import torch

input_file = "/cluster/work/projects/nn9851k/mariiaf/diachronic/nob_Latn/2024_2025_t5_nob_Latn/å_1.pt.gz"
with gzip.GzipFile(input_file, 'rb') as f:
    data = torch.load(f)
    for token_id, values in data.items():
        print(f"Lemma: {token_id}", flush=True)
        print(f"Embedding matrix size: {values[0][0].size()}", flush=True)
        print(len(values[1]), flush=True)
        print(values[1], flush=True)
        print(values[0], flush=True)
        break