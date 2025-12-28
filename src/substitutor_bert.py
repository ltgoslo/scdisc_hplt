from collections import defaultdict
import logging
import gzip
import json
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
class Substitutor:
    def __init__(self, output_dir, cache_dir, batch_size=400, language="eng_Latn", model_name=None):
        if model_name is None:
            # By default would use the T5 model pretrained on the data
            model_name = 'HPLT/hplt_bert_base_2_0_{}'.format(language)
        self.output_dir = output_dir
        self.full_model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir, use_safetensors=False,
        )
        self.full_model.eval()
        self.full_model.to('cuda') # GPU
        # Load target words
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        with open(os.path.join("../languages", language, 'target_words.json')) as f:
            target_words = json.load(f)
        tokenizer_fullwords = {
            #self.tokenizer.convert_tokens_to_string([tok]): tok_id for tok, tok_id in self.tokenizer.vocab.items() if tok.startswith("âĸģ")
            "AI": 6944, "remote": 6111, "jurisdiction": 12919, "legislative": 9935,
        }
        target_token_ids = []
        for target_word in target_words:
            if target_word in tokenizer_fullwords:
                target_token_ids.append(tokenizer_fullwords[target_word])
        print(len(target_token_ids), flush=True)
        self.target_token_ids = torch.tensor(target_token_ids).to("cuda")
        self.target_counter = defaultdict(int)
        with open(f"../languages/{language}/pos_tagged_T5_vocabulary.json", "r") as lemmas_f:
            lemmas_dict = json.load(lemmas_f)
        self.id2lemma = {}
        self.lemma2id = defaultdict(list)
        first_letters = set()
        for tok, value in lemmas_dict.items():
            if tok in tokenizer_fullwords:
                if tokenizer_fullwords[tok] in self.target_token_ids:
                    lemma = value["lemma"]
                    if language != "deu_Latn":
                        lemma = lemma.lower()
                    self.id2lemma[tokenizer_fullwords[tok]] = lemma
                    self.lemma2id[lemma].append(tokenizer_fullwords[tok])
                    first_letters.add(lemma[0])
        print(len(first_letters), flush=True)
        self.shard_files = {}
        for letter in first_letters:
            try:
                self.shard_files[letter] = gzip.open(os.path.join(self.output_dir, f"{letter}.jsonl.gz"), "wt")
            except ValueError:
                print(letter, flush=True)
                raise ValueError
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.mask_1_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.mask_1 = torch.tensor([self.mask_1_id]).to("cuda")
        self.top_k = 5
        self.batch_size = batch_size
        self.max_samples = 100
        


    def _run_prediction(self, batch, lemmas, segment_ids_batch, indices):
        logging.debug(f"Batch {batch}")
        logging.debug(f"segment_ids_batch {segment_ids_batch}")
        input_ids = pad_sequence(batch, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = (input_ids > self.pad_token_id).int()
        with torch.no_grad():
            token_logits = self.full_model(input_ids, attention_mask).logits
        for lemma, out, segment_id, idx in zip(lemmas, token_logits, segment_ids_batch, indices):
            out = out[idx]
            topk_tokens = torch.topk(out, self.top_k+10, dim=0)
            prediction = self.tokenizer.batch_decode(topk_tokens.indices.detach().cpu())
            shard_file = self.shard_files[lemma[0]]
            shard_file.write(json.dumps({lemma: [prediction, segment_id]}) + "\n")
            self.target_counter[lemma] += 1
            if self.target_counter[lemma] == self.max_samples:
                self.target_token_ids = self.target_token_ids[
                    ~torch.isin(self.target_token_ids, torch.tensor(self.lemma2id[lemma]).to("cuda"))
                ]
                print(f"Current size of target ids: {self.target_token_ids.size()}", flush=True)
        batch = []
        lemmas = []
        segment_ids_batch = []
        indices = []
        return batch, lemmas, segment_ids_batch, indices
        
        
    def process_batch(self, segments, segment_ids):
        batch = []
        lemmas = []
        segment_ids_batch = []
        indices = []
        for segment, segment_id in zip(segments, segment_ids):
            segment = segment.to("cuda")
            target_position_mask = torch.isin(segment, self.target_token_ids)
            if target_position_mask.any():
                target_indices = torch.unique(torch.where(target_position_mask, segment, 0).nonzero(as_tuple=False)[:,0], dim=0)
                logging.debug(target_indices)
                for idx in target_indices:
                    token_id = segment[idx]
                    token_id_item = token_id.item()
                    lemma = self.id2lemma[token_id_item]
                    if self.target_counter[lemma] < self.max_samples:
                        sentence = torch.cat((segment[:idx], self.mask_1, segment[idx + 1:]))
                        if len(batch) < self.batch_size:
                            batch.append(sentence)
                            lemmas.append(lemma)
                            segment_ids_batch.append(
                                (
                                self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(sentence.detach().cpu())),
                                segment_id,
                                )
                            )
                            indices.append(idx)
                        else:
                            batch, lemmas, segment_ids_batch, indices = self._run_prediction(batch, lemmas, segment_ids_batch, indices)   
                if batch:
                    batch, lemmas, segment_ids_batch, indices = self._run_prediction(batch, lemmas, segment_ids_batch, indices)
        