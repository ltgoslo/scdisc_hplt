import argparse
from collections import Counter, defaultdict
import io
import json
from glob import glob
import gzip
import os
import zstandard as zstd

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="/cluster/work/projects/nn9851k/corpora/diachronic/*/*.jsonl.zst")
parser.add_argument('--model_prefix', default="HPLT/hplt_bert_base_2_0_")
parser.add_argument('--out', default="/cluster/work/projects/nn9851k/mariiaf")
args = parser.parse_args() 

rank = int(os.environ["SLURM_PROCID"])
print(rank, flush=True)
args.input_path = glob(args.input_path)[rank]
print(args, flush=True)
exists = True
splitted = args.input_path.split('/')
lang = splitted[-2]
if "hplt" in args.model_prefix:
    langs_mapping = {'arb_Arab': 'ara_Arab'}
    model_lang = langs_mapping.get(lang, lang)
    if "bert" in args.model_prefix:
        model_lang = model_lang.replace("_", "-")
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"{args.model_prefix}{model_lang}")
    except OSError:
        print(f"no bert for {lang}")
        exists = False
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_prefix)
if exists:
    counter = defaultdict(int)
    segment_lengths = 0
    n_segments = 0
    doc_lengths = 0
    n_docs = 0
    args.out = os.path.join(args.out, "diachronic", lang)
    os.makedirs(args.out, exist_ok=True)
    output_file = os.path.join(args.out, splitted[-1].removesuffix(".jsonl.zst"))
    if not os.path.exists(output_file + '_lengths.json'):
        dctx = zstd.ZstdDecompressor()
        print(args.input_path, flush=True)        
        with gzip.open(output_file + '_tokens.jsonl.gz', 'wt') as out:
            with open(args.input_path, "rb") as f:
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    for i, line in enumerate(text_stream):
                        line = json.loads(line)
                        if line["text"]:
                            segments = line["text"].split("\n")
                            encoding = tokenizer(
                                segments,
                                add_special_tokens=False,
                                padding=False,
                                truncation=False,
                                return_length=True,
                            )
                            total_length = sum(encoding['length'])
                            segment_lengths += total_length
                            doc_lengths += total_length
                            n_docs += 1
                            n_segments += len(encoding['length'])
                            for segment_ids in encoding['input_ids']:
                                for tok, count in Counter(segment_ids).items():
                                    counter[tok] += count
                            encoding['id'] = line['id']
                            out.write(json.dumps(encoding.data) + '\n')
                        if i % 100000 == 0:
                            print(f"{output_file} {i} done", flush=True)
        with open(output_file + '_token_count.json', 'w', encoding='utf8') as f:
            json.dump(counter, f)
        averages = {
                    'average_segment_length': segment_lengths / n_segments,
                    'average_doc_length': doc_lengths / n_docs,
                }
        print(f"{output_file} {averages}", flush=True)
        with open(output_file + '_lengths.json', 'w', encoding='utf8') as f:
            json.dump(averages, f)
