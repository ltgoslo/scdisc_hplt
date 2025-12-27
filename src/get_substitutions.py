import os
import io
import json
import zstandard as zstd

import torch
from tqdm import tqdm
from optparse import OptionParser
    
from substitutor import Substitutor


def main():
    """Scrpt to process the data. By default it tokenizes sentences from the corpora. 
    Optionally, sentences can be lemmatized and tokens can be tagged for part of speech
    """
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option('--language', 
                      type=str, 
                      default='eng_Latn',
                      help='Dataset language code: default=%default')
    parser.add_option('--period',
                      type=str,
                      default='2011_2015',
                      help='Time period: default=%default')
    parser.add_option('--model-name', 
                      type=str, 
                      default="HPLT/hplt_t5_base_3_0_eng_Latn",
                      help='Model name/path to be utilized during tokenization.')
    parser.add_option('--data-dir',
                      type=str,
                      default='../data',
                      help='Default directory where diachronic data by language is stored: default=%default')
    parser.add_option('--output-dir',
                      type=str,
                      default=None,
                      help='Default directory for storing representations: default=%default')
    parser.add_option('--max-batch-size', 
                      type=int,
                      default=16,
                      help='Batch size: default=%default')
    parser.add_option('--cache-dir')
    (options, _) = parser.parse_args()
    print(options, flush=True)
    period = options.period
    assert period in {'2011_2015', '2020_2021', '2024_2025'}
    
    data_dir = options.data_dir
    language = options.language
    output_dir = '{}/{}/{}__substitutions'.format(options.output_dir, language, period)
    os.makedirs(output_dir, exist_ok=True)
    max_batch_size = options.max_batch_size
    substitutor = Substitutor(
        output_dir, options.cache_dir, max_batch_size, language, options.model_name,
    )
    
    infile = os.path.join(data_dir, language, '{}_tokens_filtered.jsonl.zst'.format(period))
    with open(infile, 'rb') as in_f:
        decompressor = zstd.ZstdDecompressor()
        stream_reader = decompressor.stream_reader(in_f)
        stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")

        segments_batch = []
        segment_ids_batch = []
        for line in tqdm(stream):
            if len(substitutor.target_token_ids) == 0:
                segment_ids_batch = []
                break
            segment_dct = json.loads(line)
            segment_tokens = torch.tensor(segment_dct['input_ids'])
            segment_id = segment_dct['s_id']
            segment_ids_batch.append(segment_id)
            segments_batch.append(segment_tokens)
            if len(segment_ids_batch) >= max_batch_size:
                substitutor.process_batch(
                    segments_batch,
                    segment_ids_batch,
                )
                segments_batch = []
                segment_ids_batch = []
                
        # process the last batch
        if len(segment_ids_batch) > 0:
            substitutor.process_batch(
                segments_batch,
                segment_ids_batch,
            )
            segments_batch = []
            segment_ids_batch = []

        for shard_file in substitutor.shard_files.values():
            shard_file.close()
        print(substitutor.target_counter, flush=True)
        with open(os.path.join(output_dir, "lemmas_counter.json"), 'w') as f:
            json.dump(substitutor.target_counter, f, ensure_ascii=False)
        

if __name__ == '__main__':
    main()