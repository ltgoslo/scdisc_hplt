import os
import io
import json
from tqdm import tqdm
import zstandard as zstd
from optparse import OptionParser
from transformers import AutoTokenizer

	

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
	parser.add_option('--data-dir',
					  type=str,
					  default='../data',
					  help='Default directory where diachronic data by language is stored: default=%default')
	parser.add_option('--tokenizer-model', 
					  type=str, 
					  default=None,
					  help='Model name/path to be utilized during tokenization.')
	parser.add_option('--batch-size', 
					  type=int,
					  default=64,
					  help='Batch size for tokenization: default=%default')
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
	data_dir = options.data_dir
	batch_size = options.batch_size

	# Load target words
	with open(os.path.join(data_dir, language, 'target_words.json')) as f:
		target_words = json.load(f)
	# with open(os.path.join(data_dir, language, 'pos_tagged_T5_vocabulary.json')) as f:
	# 	pos_tagged_T5_vocabulary = json.load(f)
	# global target_word_token_ids   
	# target_word_token_ids = [pos_tagged_T5_vocabulary[word]['id'] for word in target_words]

	infile = os.path.join(data_dir, language, '{}.jsonl.zst'.format(period))
	outfile = os.path.join(data_dir, language, '{}__filtered_segments.jsonl.zst'.format(period))

	# Load tokenizer for pretrained model
	print('Loading the tokenizer ({})...'.format(model_name))
	global tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	global frequencies_by_token_id, filtered_paragraph_count
	frequencies_by_token_id = {target_words[word]: 0 for word in target_words}
	total_doc_count, total_paragraph_count, filtered_paragraph_count = 0, 0, 0

	with open(infile, 'rb') as in_f:
		decompressor = zstd.ZstdDecompressor()
		stream_reader = decompressor.stream_reader(in_f)
		stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")
		with zstd.open(outfile, 'w') as out_f:
			# Tokenize paragraphs in batches
			paragraphs_batch = []
			paragraph_ids_batch = []
			for line in tqdm(stream, total=10**6, mininterval=60):
				# if total_paragraph_count > 300:
				# 	break
				total_doc_count += 1
				doc_dct = json.loads(line)
				paragraphs = doc_dct['text'].split('\n')
				paragraph_i = 0
				for paragraph in paragraphs:
					if len(paragraph) > 0:
						paragraph_id = '{}__{}'.format(doc_dct['id'], paragraph_i)
						paragraph_ids_batch.append(paragraph_id)
						paragraphs_batch.append(paragraph)
					if len(paragraph_ids_batch) == batch_size:
						# process batch
						input_ids = tokenizer(paragraphs_batch, padding=True, truncation=False, return_tensors="pt").input_ids
						for i in range(input_ids.shape[0]):
							paragraph_id = paragraph_ids_batch[i]
							paragraph_tokens = input_ids[i]
							target_words_found = False
							for j in range(input_ids.shape[1]):
								token_id = paragraph_tokens[j].item()
								if token_id == 0:
									break
								if token_id in frequencies_by_token_id:
									frequencies_by_token_id[token_id] += 1
									target_words_found = True
							if target_words_found:
								out_f.write(json.dumps({'s_id': paragraph_id, 
														# 'text': paragraphs_batch[i],
														'tokens': paragraph_tokens[paragraph_tokens != 0].tolist()
														}) + '\n')
							else:
								filtered_paragraph_count =+ 1
						paragraphs_batch = []
						paragraph_ids_batch = []
					paragraph_i += 1
					total_paragraph_count += 1
			input_ids = tokenizer(paragraphs_batch, padding=True, truncation=False, return_tensors="pt").input_ids
			for i in range(input_ids.shape[0]):
				paragraph_id = paragraph_ids_batch[i]
				paragraph_tokens = input_ids[i]
				target_words_found = False
				for j in range(input_ids.shape[1]):
					token_id = paragraph_tokens[j].item()
					if token_id == 0:
						break
					if token_id in frequencies_by_token_id:
						frequencies_by_token_id[token_id] += 1
						target_words_found = True
				if target_words_found:
					out_f.write(json.dumps({'s_id': paragraph_id, 
											# 'text': paragraphs_batch[i],
											'tokens': paragraph_tokens[paragraph_tokens != 0].tolist()
											}) + '\n')
				else:
					filtered_paragraph_count =+ 1
	
	# save frequency info
	with open(os.path.join(data_dir, language, '{}__frequencies_info.json'.format(period)), 'w') as f:
		json.dump(frequencies_by_token_id, f)

	with open(os.path.join(data_dir, language, '{}__paragraph_filtering_info.json'.format(period)), 'w') as f:
		json.dump({'total_doc_count': total_doc_count, 
					'total_paragraph_count':total_paragraph_count, 
					'filtered_paragraph_count': filtered_paragraph_count}, f)

if __name__ == '__main__':
	main()