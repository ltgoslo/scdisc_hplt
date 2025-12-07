import re
import io
import os
import json
import string
from tqdm import tqdm
import zstandard as zstd
from optparse import OptionParser
from transformers import AutoTokenizer

# POS-taggers
import stanza # most languages
import classla # 'mkd_Cyrl'


# Language-specific tokenizers
from fugashi import Tagger # for jpn_Jpan
import jieba # for cmn_Hans

def remove_punctuation_regex(text):
	"""
	Removes all punctuation using regular expressions.
	"""
	# Create a pattern to match any character that is a punctuation mark
	# re.escape ensures all punctuation characters are treated as literals
	pattern = '[' + re.escape(string.punctuation) + ']'
	
	# Replace all matches with an empty string
	return re.sub(pattern, ' ', text)

def space_based_word_splitter(text):
	'''Takes in a text string, removes punctuation and returns a 
	list of words that were separated by spaces'''
	return remove_punctuation_regex(text).split()

def jpn_Jpan_word_splitter(text):
	'''Takes in a text string in jpn_Jpan and returns a list of words.
	Relies on a global variable `tagger`'''
	return [str(word) for word in tagger(text)]

def cmn_Hans_word_splitter(text):
	'''Takes in a text string in and returns a list of words.
	Relies on '''
	return jieba.lcut(text)


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
	parser.add_option('--data-dir',
					  type=str,
					  default='../data',
					  help='Default directory where diachronic data by language is stored: default=%default')
	parser.add_option('--pos-tagger-resources-dir', 
					  type=str, 
					  default='~/stanza_resources',
					  help='Dataset language code: default=%default')
	parser.add_option('--pos-to-keep',
					  type=str,
					  default='NOUN|VERB|ADJ',
					  help='Parts-of-speech to keep in the vocabulary, separated by "|": default=%default')
	parser.add_option('--word-freq-threshold',
					  type=int,
					  default=2,
					  help='The minimum number of times the token should appear as a full word: default=%default')
	# parser.add_option('--verbose',
				# 	  action='store_true',
				# 	  help='Whether to print out more detailed info: default=%default')

	(options, args) = parser.parse_args()
	global language
	language = options.language
	data_dir = options.data_dir
	pos_tagger_resources_dir = options.pos_tagger_resources_dir
	pos_to_keep = options.pos_to_keep.split('|')
	word_freq_threshold = options.word_freq_threshold

	periods = ['2011_2015', '2020_2021', '2024_2025']
	infile = os.path.join(data_dir, language, '{}.jsonl.zst')

	# Load the tokenizer
	model_name = 'HPLT/hplt_t5_base_3_0_{}'.format(language)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	# Load some additional info about language codes
	with open('language_code_to_name_mapping.json') as f:
		language_mapping_info = json.load(f)
	assert language in language_mapping_info

	# Load a word-splitter 
	if language in ['tha_Thai', 'vie_Latn']:
		raise ExceptionClass("Language {} word-splitting not implemented yet!".format(language))
	elif language == 'jpn_Jpan':
		global tagger
		tagger = Tagger('-Owakati')
		word_splitter = jpn_Jpan_word_splitter
	elif language == 'cmn_Hans':
		word_splitter = cmn_Hans_word_splitter
	elif language in language_mapping_info:
		word_splitter = space_based_word_splitter
	else:
		raise ExceptionClass("Unknown language!")

	# Do a word-splitting pass through the data
	full_word_frequencies = {}
	for period in periods:
		with open(infile.format(period), 'rb') as f:
			# open the data file
			decompressor = zstd.ZstdDecompressor()
			stream_reader = decompressor.stream_reader(f)
			stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")
			# make a pass through lines, splitting words
			for line in tqdm(stream, total=10**6, desc='Processing lines from {} period'.format(period), mininterval=30):
				for word in word_splitter(json.loads(line)['text']):
					word_lower = word.lower()
					if word_lower not in full_word_frequencies:
						full_word_frequencies[word_lower] = 0
					full_word_frequencies[word_lower] += 1

	with open(os.path.join(data_dir, language, 'full_word_frequencies.json'), 'w') as f:
		json.dump(full_word_frequencies, f)


	# Load the pos-tagger
	if language in ['als_Latn', 'bos_Latn', 'kat_Geor', 'tha_Thai']:
		raise ExceptionClass("Part-of-speech tagger not integrated yet!")
	elif language == 'mkd_Cyrl':
		nlp = classla.Pipeline('mk', dir=pos_tagger_resources_dir)
	else:
		nlp = stanza.Pipeline(language_mapping_info[language][1], model_dir=pos_tagger_resources_dir)
		
	# run T5 vocabulary through a POS-tagger and only keep needed parts-of-speech
	pos_tagged_T5_vocabulary = {}
	for elt in tqdm(tokenizer.vocab, mininterval=30):
		token_id = tokenizer.vocab[elt]
		vocab_token = tokenizer.convert_tokens_to_string([elt])
		processed_token = nlp(vocab_token)
		if len(processed_token.sentences) > 0 and len(processed_token.sentences[0].words) > 0:
			parsed_vocab_token = processed_token.sentences[0].words[0]
			pos_tagged_T5_vocabulary[parsed_vocab_token.text] = {'id': token_id,
																 'lemma': parsed_vocab_token.lemma,
																 'upos': parsed_vocab_token.upos}

	# check NOUNs: does POS-tag match across lower-cased & capitlized?
	for token in pos_tagged_T5_vocabulary:
		if pos_tagged_T5_vocabulary[token]['upos'] == 'PROPN' \
		and token.lower() in pos_tagged_T5_vocabulary \
		and pos_tagged_T5_vocabulary[token.lower()]['upos'] == 'NOUN':
			pos_tagged_T5_vocabulary[token]['upos'] = 'NOUN'
			
	# Once refined, the full-word filtering code would go here, but for now it is taking too long to run				
	with open(os.path.join(data_dir, language, 'pos_tagged_T5_vocabulary.json'), 'w') as f:
		json.dump(pos_tagged_T5_vocabulary, f)

	# Filter T5 vocabulary to obtain Target Words
	target_words = []
	for token in pos_tagged_T5_vocabulary:
		if pos_tagged_T5_vocabulary[token]['upos'] in pos_to_keep and full_word_frequencies.get(token.lower(), 0) > word_freq_threshold:
			target_words.append(token)

	target_words = sorted(target_words)
	with open(os.path.join(data_dir, language, 'target_words.json'), 'w') as f:
		json.dump(target_words, f)

				

if __name__ == '__main__':
	main()














