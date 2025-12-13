import os
import io
import json
import pickle
import zstandard as zstd

from tqdm import tqdm
from optparse import OptionParser

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import save_file
	

def save_embeddings_packet(token_id):
	global embedding_packet_data, embedding_packet_info, embedding_packet_size, embedding_packet_count

	# create the output directory if missing
	token_out_dir = os.path.join(embeddings_dir, str(token_id))
	if not os.path.exists(token_out_dir):
		os.makedirs(token_out_dir)
	current_packet_number = embedding_packet_count

	# save the embeddings packet
	content = {str(current_packet_number): embedding_packet_data[:embedding_packet_size, ]}
	save_file(content, os.path.join(token_out_dir, "packet_{}.safetensors".format(current_packet_number)))
	
	# update packet information for this
	if current_packet_number > 0:
		with open(os.path.join(token_out_dir, "packets_info.json")) as f:
			current_packet_info = json.load(f)
	else:
		current_packet_info = {}
	new_key = "packet_{}".format(current_packet_number) 
	assert new_key not in current_packet_info
	current_packet_info[new_key] = embedding_packet_info
	with open(os.path.join(token_out_dir, "packets_info.json"), 'w') as f:
		json.dump(current_packet_info, f)

	# reset/update packet information in our global variables
	embedding_packet_data = torch.empty(max_packet_size, 768, dtype=torch.float32)
	embedding_packet_info = []
	embedding_packet_size = 0
	embedding_packet_count += 1
	

def process_batch(segments_batch, segment_ids_batch, token_id):
	global embedding_packet_data, embedding_packet_info, embedding_packet_size
	
	# pad the batch of tokens and re-create attention mask
	input_ids = pad_sequence(segments_batch, batch_first=True, padding_value=0).to("cuda")
	attention_mask = (input_ids > 0).int()
	# print('[process batch] input_ids shape : {}'.format(input_ids.shape))

	# obtain position mask for the token_id of the given target word
	target_position_mask_all = (input_ids == token_id)
	target_position_mask_cum = target_position_mask_all.cumsum(dim=-1)
	target_position_mask_first = (target_position_mask_cum == 1) & target_position_mask_all

	# get embeddings of the batch
	with torch.no_grad():
		outputs = encoder_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
	embeddings_batch = outputs.last_hidden_state[target_position_mask_first].detach().cpu()

	batch_size = embeddings_batch.shape[0]
	if (embedding_packet_size + batch_size) > max_packet_size:
		save_embeddings_packet(token_id)
	# print(embeddings_batch.shape)
	embedding_packet_data[embedding_packet_size:(embedding_packet_size+batch_size)] = embeddings_batch
	embedding_packet_info += segment_ids_batch
	embedding_packet_size += batch_size

	# # for each paragraph in the batch
	# for i in range(batch_size):
	# 	paragraph_id = paragraph_ids_batch[i]
	# 	# go over each token in the paragraph
	# 	for j in range(batch_token_length):
	# 		token_id = input_ids[i,j].item()
	# 		if token_id in target_word_token_ids:
	# 			# add that token's embedding to the corresponding row in its packet
	# 			token_embedding = embeddings_batch[i,j,:].detach().cpu()
	# 			current_packet_size = len(packet_info_by_token_id[token_id])
	# 			embedding_packets_by_token_id[token_id][current_packet_size] = token_embedding
	# 			current_packet_size += 1
	# 			packet_info_by_token_id[token_id].append(paragraph_id)
	# 			frequencies_by_token_id[token_id] += 1
	# 			if current_packet_size >= max_packet_size:
	# 				save_embeddings_packet(token_id)
	

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
					  default=None,
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
					  help='Batch size for tokenization: default=%default')
	parser.add_option('--packet-size-multiplier-N', 
					  type=int,
					  default=100,
					  help='N that would be multiplied by max-batch-size to obtain the packet size for word embedding packets: default=%default')
	# parser.add_option('--verbose',
				# 	  action='store_true',
				# 	  help='Whether to print out more detailed info: default=%default')

	(options, args) = parser.parse_args()

	language = options.language
	period = options.period
	assert period in ['2011_2015', '2020_2021', '2024_2025']
	model_name = options.model_name
	if model_name is None:
		# By default would use the T5 model pretrained on the data
		model_name = 'HPLT/hplt_t5_base_3_0_{}'.format(language)
	data_dir = options.data_dir
	global embeddings_dir
	embeddings_dir = options.embeddings_dir
	if embeddings_dir is None:
		embeddings_dir = '../representations/{}/{}__embeddings'.format(language, period)
	if not os.path.exists(embeddings_dir):
		os.makedirs(embeddings_dir)
	max_batch_size = options.max_batch_size
	global max_packet_size
	max_packet_size = options.packet_size_multiplier_N * max_batch_size

	# Load target words
	with open(os.path.join(data_dir, language, 'target_words.json')) as f:
		target_words = json.load(f)
	
	infile = os.path.join(data_dir, language, '{}__filtered_segments.jsonl.zst'.format(period))

	# Load tokenizer for pretrained model
	print('Loading the tokenizer ({})...'.format(model_name))
	global tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, use_safetensors=False)
	full_model.eval()
	full_model.to('cuda') # GPU
	global encoder_model
	encoder_model = full_model.get_encoder()
	print("Model loaded:", model_name)

	# global frequencies_by_token_id, embedding_packets_by_token_id, packet_info_by_token_id, packets_count_by_token_id
	# frequencies_by_token_id = {target_words[word]: 0 for word in target_words}
	# embedding_packets_by_token_id = {target_words[word]: torch.empty(max_packet_size, 768, dtype=torch.float32) for word in target_words}
	# packet_info_by_token_id = {target_words[word]: [] for word in target_words}
	# packets_count_by_token_id = {target_words[word]: 0 for word in target_words}
	
	global embedding_packet_data, embedding_packet_info, embedding_packet_size, embedding_packet_count
	# global frequencies_by_token_id
	# frequencies_by_token_id = {target_words[word]: 0 for word in target_words}

	# Go through the lines of the file
	for word in tqdm(target_words):
		token_id = target_words[word]
		embedding_packet_data = torch.empty(max_packet_size, 768, dtype=torch.float32)
		embedding_packet_info = []
		embedding_packet_size = 0
		embedding_packet_count = 0
		with open(infile, 'rb') as in_f:
			decompressor = zstd.ZstdDecompressor()
			stream_reader = decompressor.stream_reader(in_f)
			stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")

			# Embed segments in batches
			segments_batch = []
			segment_ids_batch = []
			# for line in tqdm(stream, total=10**6, mininterval=60):
			for line in stream:
				# if embedding_packet_count > 5:
				# 	break
				segment_dct = json.loads(line)
				# NEEDS UPDATED TRUNCATION
				segment_tokens = torch.tensor(segment_dct['tokens'][:512])
				segment_id = segment_dct['s_id']
				token_presence_mask = (segment_tokens == token_id)
				if token_presence_mask.any().item():
					segment_ids_batch.append(segment_id)
					segments_batch.append(segment_tokens)
					if len(segment_ids_batch) >= max_batch_size:
						process_batch(segments_batch, segment_ids_batch, token_id)
						segments_batch = []
						segment_ids_batch = []
			# process the last batch
			if len(segment_ids_batch) > 0:
				process_batch(segments_batch, segment_ids_batch, token_id)
				segments_batch = []
				segment_ids_batch = []

			# save the remaining embedding packet
			if embedding_packet_size > 0:
				save_embeddings_packet(token_id)
			# break


	# # go through and save all the remaining embedding packets
	# for token_id in packet_info_by_token_id:
	# 	current_packet_size = len(packet_info_by_token_id[token_id])
	# 	if current_packet_size > 0:
	# 		sliced_packet = embedding_packets_by_token_id[token_id][:current_packet_size] 
	# 		embedding_packets_by_token_id[token_id] = sliced_packet
	# 		save_embeddings_packet(token_id)

	# # save frequency info
	# with open(os.path.join(data_dir, langauge, '{}__frequencies_info.json'.format(period)), 'w') as f:
	# 	json.dump(frequencies_by_token_id, f)


if __name__ == '__main__':
	main()