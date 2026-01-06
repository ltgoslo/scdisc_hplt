import re
import io
import os
import json
import string
from tqdm import tqdm
import zstandard as zstd
from optparse import OptionParser
from transformers import AutoTokenizer
import unicodedata as ud
import warnings
warnings.filterwarnings("ignore")

# POS-taggers
import stanza # most languages
import classla # 'mkd_Cyrl'


# Language-specific tokenizers
from fugashi import Tagger # for jpn_Jpan
import jieba # for cmn_Hans
import pythainlp

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

def tha_Thai_word_splitter(text):
	return pythainlp.word_tokenize(text)

def verify_script(word, script_code):
	script_code_to_name = {'Arab': 'ARABIC',
						 'Armn': 'ARMENIAN',
						 'Cyrl': 'CYRILLIC',
						 'Geor': 'GEORGIAN',
						 'Grek': 'GREEK',
						 'Hang': 'HANGUL',
						 'Hans': 'CJK',
						 'Hebr': 'HEBREW',
						 'Jpan': ['CJK', 'HIRAGANA', 'KATAKANA'],
						 'Latn': 'LATIN',
						 'Taml': 'TAMIL',
						 'Thai': 'THAI'}
	for char in word:
		if char not in seen_character_status:
			char_full_script_name = ud.name(char)
			if script_code == 'Jpan':
				if 'CJK' not in char_full_script_name \
				and 'HIRAGANA' not in char_full_script_name \
				and 'KATAKANA' not in char_full_script_name:
					seen_character_status[char] = False
				else:
					seen_character_status[char] = True
			else:
				seen_character_status[char] = (script_code_to_name[script_code] in char_full_script_name)
		if not seen_character_status[char]:
			return False
	return True



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
	parser.add_option('--git-dir',
					  type=str,
					  default='../scdisc_hplt/languages',
					  help='Git directory where to output `target_words.json` and `pos_tagged_T5_vocabulary.json`: default=%default')
	parser.add_option('--pos-tagger-resources-dir', 
					  type=str, 
					  default='~/stanza_resources',
					  help='Dataset language code: default=%default')
	parser.add_option('--pos-to-keep',
					  type=str,
					  default='NOUN|VERB|ADJ',
					  help='Parts-of-speech to keep in the vocabulary, separated by "|": default=%default')
	parser.add_option('--inter-period-min-freq',
					  type=int,
					  default=10,
					  help='The minimum number of times the token should appear as a full word across periods: default=%default')
	parser.add_option('--intra-period-min-freq',
					  type=int,
					  default=3,
					  help='The minimum number of times the token should appear as a full word within each period: default=%default')
	parser.add_option('--not-compute-full-word-freqs',
					  action='store_true',
					  help='Whether to recompute full word frequencies. If not loads frequencies from\
					  `{data_dir}/{language}/full_word_frequencies.json`: default=%default')
	parser.add_option('--not-run-pos-tagging',
					  action='store_true',
					  help='Whether rerun part-of-speech tagging. If not loads part-of-speech in from\
					  `{data_dir}/{language}/pos_tagged_T5_vocabulary.json`: default=%default')

	# parser.add_option('--verbose',
				# 	  action='store_true',
				# 	  help='Whether to print out more detailed info: default=%default')

	(options, args) = parser.parse_args()
	global language
	language = options.language
	# if language == 'ara_Arab':
	# 	language = 'arb_Arab'
	data_dir = options.data_dir
	git_dir = options.git_dir
	pos_tagger_resources_dir = options.pos_tagger_resources_dir
	pos_to_keep = options.pos_to_keep.split('|')
	inter_period_min_freq = options.inter_period_min_freq
	intra_period_min_freq = options.intra_period_min_freq
	not_compute_full_word_freqs = options.not_compute_full_word_freqs
	not_run_pos_tagging = options.not_run_pos_tagging

	periods = ['2011_2015', '2020_2021', '2024_2025']
	infile = os.path.join(data_dir, language, '{}.jsonl.zst')

	# Load the tokenizer
	if language == 'arb_Arab':
		model_name = 'HPLT/hplt_t5_base_3_0_{}'.format('ara_Arab')
		tokenizer = AutoTokenizer.from_pretrained(model_name)
	else:
		model_name = 'HPLT/hplt_t5_base_3_0_{}'.format(language)
		tokenizer = AutoTokenizer.from_pretrained(model_name)

	# Load some additional info about language codes
	with open('language_code_to_name_mapping.json') as f:
		language_mapping_info = json.load(f)
	assert language in language_mapping_info

	# Load a word-splitter 
	if language == 'tha_Thai':
		word_splitter = tha_Thai_word_splitter
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
	if not_compute_full_word_freqs:
		with open(os.path.join(data_dir, language, 'full_word_frequencies.json')) as f:
			full_word_frequencies = json.load(f)
	else:
		full_word_frequencies = {}
		for i in range(len(periods)):
			period = periods[i]
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
							full_word_frequencies[word_lower] = [0, 0, 0]
						full_word_frequencies[word_lower][i] += 1

		with open(os.path.join(data_dir, language, 'full_word_frequencies.json'), 'w') as f:
			json.dump(full_word_frequencies, f)


	if not_run_pos_tagging:
		with open(os.path.join(data_dir, language, 'pos_tagged_T5_vocabulary.json')) as f:
			pos_tagged_T5_vocabulary = json.load(f)
	else:

		# Load the pos-tagger
		if language == 'bos_Latn':
			nlp = stanza.Pipeline('hr', model_dir=pos_tagger_resources_dir)
		elif language == 'mkd_Cyrl':
			nlp = classla.Pipeline('mk', dir=pos_tagger_resources_dir)
		elif language_mapping_info[language][1] is not None:
			nlp = stanza.Pipeline(language_mapping_info[language][1], processors="tokenize,pos,lemma", model_dir=pos_tagger_resources_dir)
		else:
			raise ExceptionGroup("Part-of-speech tagger not integrated yet!")

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
		with open(os.path.join(git_dir, language, 'pos_tagged_T5_vocabulary.json'), 'w') as f:
			json.dump(pos_tagged_T5_vocabulary, f)


	# Filter T5 vocabulary to obtain Target Words
	global seen_character_status
	seen_character_status = {}
	script_code = language.split('_')[-1]
	target_words = {}
	single_char_tokens_count = 0
	full_word_freq_filtered_count = 0
	for token in tqdm(pos_tagged_T5_vocabulary):
		frequencies = full_word_frequencies.get(token.lower())
		# part-of-speech filtering (NOUNs, VERBs, and ADJectives)
		if pos_tagged_T5_vocabulary[token]['upos'] in pos_to_keep and frequencies is not None:
			# frequency filtering (inter & intra)
			if sum(frequencies) >= inter_period_min_freq and min(frequencies) >= intra_period_min_freq:
				# script filtering (all characters should be part of the given script)
				if verify_script(token, script_code):
					# filter out one-character words (except in cmn_Hans, )
					if language in ['cmn_Hans', 'jpn_Jpan', 'kor_Hang']:
						target_words[token] = pos_tagged_T5_vocabulary[token]['id']
					elif len(token) > 1:
						target_words[token] = pos_tagged_T5_vocabulary[token]['id']
					else:
						single_char_tokens_count += 1
			else:
				if verify_script(token, script_code) and (language in ['cmn_Hans', 'jpn_Jpan', 'kor_Hang'] or len(token) > 1):
					full_word_freq_filtered_count += 1

	print('New vocab size:', len(target_words))
	print('# of words that were filtered out using full word frequencies', full_word_freq_filtered_count)
	print('# of filtered-out single-character words:', single_char_tokens_count)

	with open(os.path.join(data_dir, language, 'target_words.json'), 'w') as f:
		json.dump(target_words, f)
	with open(os.path.join(git_dir, language, 'target_words.json'), 'w') as f:
		json.dump(target_words, f)

				

if __name__ == '__main__':
	main()
