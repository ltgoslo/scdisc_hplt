import os
import re
import json
import stanza
import classla
from tqdm import tqdm
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
	parser.add_option('--pos-tagger-resources-dir', 
					  type=str, 
					  default='~/stanza_resources',
					  help='Dataset language code: default=%default')
	parser.add_option('--pos-to-keep',
					  type=str,
					  default='NOUN|VERB|ADJ',
					  help='Parts-of-speech to keep in the vocabulary, separated by "|": default=%default')
	# parser.add_option('--verbose',
				# 	  action='store_true',
				# 	  help='Whether to print out more detailed info: default=%default')

	(options, args) = parser.parse_args()
	language = options.language
	pos_tagger_resources_dir = options.pos_tagger_resources_dir
	pos_to_keep = options.pos_to_keep.split('|')

	# Load the tokenizer
	model_name = 'HPLT/hplt_t5_base_3_0_{}'.format(language)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	# Load some additional info about language codes
	with open('../language_code_to_name_mapping.json') as f:
		language_mapping_info = json.load(f)
	assert language in language_mapping_info
		
	# Load the pos-tagger
	if language_mapping_info[language][1] is not None:
		nlp = stanza.Pipeline(filtered_language_code_to_name_mapping[language][1], model_dir=pos_tagger_resources_dir)
	elif language == 'mkd_Cyrl':
		nlp = classla.Pipeline('mk', dir=pos_tagger_resources_dir)
	else:
		raise ExceptionClass("Part-of-speech tagger not found")
	
	# run T5 vocabulary through a POS-tagger and only keep needed parts-of-speech
	pos_filtered_T5_vocabulary = {}
	for elt in tqdm(tokenizer.vocab, mininterval=30):
		token_id = tokenizer.vocab[elt]
		vocab_token = tokenizer.convert_tokens_to_string([elt])
		processed_token = nlp(vocab_token)
		if len(processed_token.sentences) > 0 and len(processed_token.sentences[0].words) > 0:
			parsed_vocab_token = processed_token.sentences[0].words[0]
			if parsed_vocab_token.upos in pos_to_keep:
				pos_filtered_T5_vocabulary[parsed_vocab_token.text] = {'id': token_id,
																	   'lemma': parsed_vocab_token.lemma,
																	   'upos': parsed_vocab_token.upos}
		
	# Once refined, the full-word filtering code would go here, but for now it is taking too long to run				
	with open('../data/{}/pos_filtered_T5_vocabulary.json'.format(language), 'w') as f:
		json.dump(pos_filtered_T5_vocabulary, f)
				


if __name__ == '__main__':
	main()














