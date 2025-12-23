from collections import defaultdict
import os
import io
import json
import zstandard as zstd

import torch
from tqdm import tqdm
from optparse import OptionParser
    
from embedder import Embedder


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
    parser.add_option('--embeddings-dir',
                      type=str,
                      default=None,
                      help='Default directory for storing embedding representations: default=%default')
    parser.add_option('--max-batch-size', 
                      type=int,
                      default=16,
                      help='Batch size: default=%default')
    parser.add_option('--max-packet-size', 
                      type=int,
                      default=2e+9,
                      help='The packet size for word embedding packets: default=%default')
    (options, args) = parser.parse_args()

    period = options.period
    assert period in {'2011_2015', '2020_2021', '2024_2025'}
    
    data_dir = options.data_dir
    embeddings_dir = options.embeddings_dir
    if embeddings_dir is None:
        embeddings_dir = '../representations/{}/{}__embeddings'.format(language, period)

    language = options.language
    max_batch_size = options.max_batch_size
    embedder = Embedder(
        embeddings_dir, max_batch_size, language, options.max_packet_size, options.model_name,
    )
    
    infile = os.path.join(data_dir, language, '{}_tokens_filtered.jsonl.zst'.format(period))
    with open(infile, 'rb') as in_f:
        decompressor = zstd.ZstdDecompressor()
        stream_reader = decompressor.stream_reader(in_f)
        stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")

        # Embed segments in batches
        segments_batch = []
        segment_ids_batch = []
        embedding_packet_data = {token_id.item(): [[], []] for token_id in embedder.target_token_ids}
        embedding_packet_size = 0
        for line in tqdm(stream):
            if len(embedder.target_token_ids) == 0:
                segment_ids_batch = []
                break
            segment_dct = json.loads(line)
            segment_tokens = torch.tensor(segment_dct['input_ids'])
            segment_id = segment_dct['s_id']
            segment_ids_batch.append(segment_id)
            segments_batch.append(segment_tokens)
            if len(segment_ids_batch) >= max_batch_size:
                embedding_packet_data, embedding_packet_size = embedder.process_batch(
                    segments_batch,
                    segment_ids_batch,
                    embedding_packet_data,
                    embedding_packet_size,
                )
                segments_batch = []
                segment_ids_batch = []
        # process the last batch
        if len(segment_ids_batch) > 0:
            embedding_packet_data, _ = embedder.process_batch(
                segments_batch,
                segment_ids_batch,
                embedding_packet_data,
                embedding_packet_size,
            )
            segments_batch = []
            segment_ids_batch = []

        # save the remaining embedding packet
        if len(embedding_packet_data) > 0:
            embedder.save_embeddings_packet(embedding_packet_data)


if __name__ == '__main__':
    main()