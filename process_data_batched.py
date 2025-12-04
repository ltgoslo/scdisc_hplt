import os
import io
import json
from tqdm import tqdm
import zstandard as zstd
from optparse import OptionParser
from transformers import AutoTokenizer


def process_batch(paragraphs_batch, paragraph_ids_batch, batch_size=64):
	indexing_lines = []
	input_ids = tokenizer(paragraphs_batch, padding=True, truncation=False, return_tensors="pt").input_ids
	for i in range(batch_size):
		paragraph_id = paragraph_ids_batch[i]
		paragraph_tokens = input_ids[i]
		j = 0
		token_id = paragraph_tokens[j].item()
		while token_id > 0 and j < len(paragraph_tokens):
			token_id = paragraph_tokens[j].item()
			if token_id in frequencies_by_token_id:
				indexing_lines.append(json.dumps({'token': token_id, 'paragraph_id': paragraph_id, 'index': j}))
				frequencies_by_token_id[token_id] += 1
			j += 1
	return indexing_lines

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
	parser.add_option('--tokenizer-model', 
					  type=str, 
					  default=None,
					  help='Model name/path to be utilized during tokenization.')
	parser.add_option('--batch-size', 
					  type=int,
					  default=64,
					  help='Batch size for tokenization: default=%default')
	parser.add_option('--break-point',
					  type=int,
					  default=5*(10**5),
					  help='When to stop the iteration through period: default: %default')
	# parser.add_option('--verbose',
				# 	  action='store_true',
				# 	  help='Whether to print out more detailed info: default=%default')

	(options, args) = parser.parse_args()

	language = options.language
	period = options.period
	assert period in ['2011_2015', '2020_2021', '2024_2025']
	model_name = options.tokenizer_model
	if model_name is None:
		# By default would use the T5 model pretrained on the data
		model_name = 'HPLT/hplt_t5_base_3_0_{}'.format(language)
	batch_size = options.batch_size
	break_point = options.break_point

	with open('../data/{}/pos_filtered_T5_vocabulary.json'.format(language)) as f:
		pos_filtered_T5_vocabulary = json.load(f)

	infile = '../data/{}/{}.jsonl.zst'.format(language, period)

	outdir = os.path.join('../data', language, 'processed__' + model_name.split('/')[-1])
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	# Load tokenizer for pretrained model
	print('Loading the tokenizer ({})...'.format(model_name))
	global tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	global frequencies_by_token_id
	frequencies_by_token_id = {id_: 0 for string, id_ in tokenizer.vocab.items()
	                        if tokenizer.convert_tokens_to_string([string]) in pos_filtered_T5_vocabulary}

	with open(infile, 'rb') as in_f:
	    decompressor = zstd.ZstdDecompressor()
	    stream_reader = decompressor.stream_reader(in_f)
	    stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")

	    # Tokenize paragraphs in batches
	    with open(os.path.join(outdir, '{}.jsonl'.format(period)), 'w') as out_f:
		    paragraphs_batch = []
		    paragraph_ids_batch = []
		    line_count = 0
		    for line in tqdm(stream, total=5*(10**5), mininterval=60):
		        line_count += 1
		        # TO COMMENT-OUT
		        if line_count > break_point:
		            break
		        doc_dct = json.loads(line)
		        paragraphs = doc_dct['text'].split('\n')
		        paragraph_i = 0
		        for paragraph in paragraphs:
		            paragraph_id = '{}__{}'.format(doc_dct['id'], paragraph_i)
		            paragraph_ids_batch.append(paragraph_id)
		            paragraphs_batch.append(paragraph)
		            if len(paragraph_ids_batch) == batch_size:
		                # process_batch(paragraphs_batch, paragraph_ids_batch, batch_size=batch_size)
		                for line_out in process_batch(paragraphs_batch, paragraph_ids_batch, batch_size=batch_size):
		                	out_f.write(line_out + '\n')	
		                paragraphs_batch = []
		                paragraph_ids_batch = []
		            paragraph_i += 1
		    for line_out in process_batch(paragraphs_batch, paragraph_ids_batch, batch_size=batch_size):
		        out_f.write(line_out + '\n')	

	with open(os.path.join(outdir, 'frequencies_info.json'.format(period)), 'w') as f:
		json.dump(frequencies_by_token_id)

if __name__ == '__main__':
	main()