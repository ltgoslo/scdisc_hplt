from collections import defaultdict
import gzip
import json
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Substitutor:
    def __init__(self, output_dir, cache_dir, batch_size=400, language="eng_Latn", model_name=None):
        if model_name is None:
            # By default would use the T5 model pretrained on the data
            model_name = 'HPLT/hplt_t5_base_3_0_{}'.format(language)
        self.output_dir = output_dir
        self.full_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, use_safetensors=False, cache_dir=cache_dir,
        )
        self.full_model.eval()
        self.full_model.to('cuda') # GPU
        # Load target words
        with open(os.path.join("../languages", language, 'target_words.json')) as f:
            target_words = json.load(f)
        self.target_token_ids = torch.tensor([token_id for token_id in target_words.values()]).to("cuda")
        self.target_counter = defaultdict(int)
        with open(f"../languages/{language}/pos_tagged_T5_vocabulary.json", "r") as lemmas_f:
            lemmas_dict = json.load(lemmas_f)
        self.id2lemma = {}
        self.lemma2id = defaultdict(list)
        first_letters = set()
        for value in lemmas_dict.values():
            lemma = value["lemma"]
            if lemma[0] == '/':
                lemma = lemma.lstrip("/")
            if value["id"] in self.target_token_ids:
                if language != "deu_Latn":
                    lemma = lemma.lower()
                self.id2lemma[value["id"]] = lemma
                self.lemma2id[lemma].append(value["id"])
                first_letters.add(lemma[0])
        print(len(first_letters), flush=True)
        self.shard_files = {}
        for letter in first_letters:
            try:
                self.shard_files[letter] = gzip.open(os.path.join(self.output_dir, f"{letter}.jsonl.gz"), "wt")
            except ValueError:
                print(letter, flush=True)
                raise ValueError
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.mask_1_id = self.tokenizer.convert_tokens_to_ids("[MASK_1]")
        self.mask_2_id = self.tokenizer.convert_tokens_to_ids("[MASK_2]")
        self.mask_1 = torch.tensor([self.mask_1_id]).to("cuda")
        self.batch_size = batch_size


    def _run_prediction(self, batch, lemmas, segment_ids_batch):
        input_ids = pad_sequence(batch, batch_first=True, padding_value=self.pad_token_id)
        output_tensor = self.full_model.generate(
            input_ids,
            decoder_start_token_id=self.mask_1_id,
            eos_token_id=self.mask_2_id,
        )

        for lemma, out, segment_id in zip(lemmas, output_tensor, segment_ids_batch):
            prediction = self.tokenizer.decode(out.detach().cpu(), skip_special_tokens=True)
            shard_file = self.shard_files[lemma[0]]
            shard_file.write(json.dumps({lemma: [prediction, segment_id]}) + "\n")
            self.target_counter[lemma] += 1
            if self.target_counter[lemma] == 1000:
                self.target_token_ids = self.target_token_ids[
                    ~torch.isin(self.target_token_ids, torch.tensor(self.lemma2id[lemma]).to("cuda"))
                ]
        batch = []
        lemmas = []
        segment_ids_batch = []
        return batch, lemmas, segment_ids_batch
        
        
    def process_batch(self, segments, segment_ids):
        batch = []
        lemmas = []
        segment_ids_batch = []
        for segment, segment_id in zip(segments, segment_ids):
            segment = segment.to("cuda")
            target_position_mask = torch.isin(segment, self.target_token_ids)
            if target_position_mask.any():
                target_indices = torch.unique(torch.where(target_position_mask, segment, 0).nonzero(as_tuple=False)[:,0], dim=0)
                for idx in tqdm(target_indices):
                    token_id = segment[idx]
                    token_id_item = token_id.item()
                    lemma = self.id2lemma[token_id_item]
                    if self.target_counter[lemma] < 1000:
                        sentence = torch.cat((segment[:idx], self.mask_1, segment[idx + 1:]))
                        if len(batch) < self.batch_size:
                            batch.append(sentence)
                            lemmas.append(lemma)
                            segment_ids_batch.append(segment_id)
                        else:
                            batch, lemmas, segment_ids_batch = self._run_prediction(batch, lemmas, segment_ids_batch)   
                if batch:
                    self._run_prediction(batch, lemmas, segment_ids_batch)
        