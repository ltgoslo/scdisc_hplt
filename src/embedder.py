from collections import defaultdict
from copy import copy
import json
import gzip
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence


class Embedder:
    def __init__(self, embeddings_dir, batch_size, language, max_packet_size, model_name=None, cache_dir="~/.cache/huggingface/"):
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)
        self.embeddings_dir = embeddings_dir
        self.embedding_size = 3072
        self.batch_size = batch_size
        self.max_packet_size = max_packet_size
        self.embedding_packet_count = 0

        if model_name is None:
            # By default would use the T5 model pretrained on the data
            model_name = 'HPLT/hplt_t5_base_3_0_{}'.format(language)
        # Load tokenizer for pretrained model
        print('Loading the tokenizer ({})...'.format(model_name), flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
        full_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, use_safetensors=False, cache_dir=cache_dir,
        )
        full_model.eval()
        full_model.to('cuda') # GPU
        self.encoder_model = full_model.get_encoder()
        print(f"Model loaded: {model_name}", flush=True)
        # Load target words
        updated = os.path.join(embeddings_dir, "target_ids.pt.gz")
        if not os.path.exists(updated):
            with open(os.path.join("../languages", language, 'target_words.json')) as f:
                target_words = json.load(f)
            self.target_token_ids = torch.tensor([token_id for token_id in target_words.values()]).to("cuda")
        else:
            with gzip.GzipFile(updated, 'rb') as f:
                self.target_token_ids = torch.load(f).to("cuda")

        self.target_counter = defaultdict(int)
        with open(f"../languages/{language}/pos_tagged_T5_vocabulary.json", "r") as lemmas_f:
            lemmas_dict = json.load(lemmas_f)
        self.id2lemma = {}
        self.lemma2id = defaultdict(list)

        for tok, value in lemmas_dict.items():
            lemma = value["lemma"]
            if lemma is None:
                lemma = copy(tok)
            if language != "deu_Latn":
                lemma = lemma.lower()
            self.id2lemma[value["id"]] = lemma
            self.lemma2id[lemma].append(value["id"])
        self.language = language
        self.save_by_size = self.language in {"cmn_Hans", "jpn_Jpan", "kor_Hang"}


    def save_embeddings_packet(self, embedding_packet_data):
        self.embedding_packet_count += 1
        for letter, letter_data in embedding_packet_data.items():
            with gzip.GzipFile(os.path.join(self.embeddings_dir, f"{letter}_{self.embedding_packet_count}.pt.gz"), 'wb') as fout:
                torch.save(letter_data, fout)
        return {}, 0
    

    def _process_embedding_packet_data(self, embeddings_batch, embedding_packet_data):
        first_letter = copy(self.language)
        for lemma in embeddings_batch.keys():
            if not self.save_by_size:
                first_letter = lemma[0].lower() # no need to do anything special for Arabic and Hebrew
                # see https://unicode.org/reports/tr9/
            if embedding_packet_data.get(first_letter) is None:
                embedding_packet_data[first_letter] = {}
            if embedding_packet_data[first_letter].get(lemma) is None:
                embedding_packet_data[first_letter][lemma] = [[torch.zeros(1, 768)], []]
            embedding_packet_data[first_letter][lemma][0] = [
                torch.cat(embedding_packet_data[first_letter][lemma][0] + embeddings_batch[lemma][0], 0),
            ]
            embedding_packet_data[first_letter][lemma][1].extend(embeddings_batch[lemma][1])
        return embedding_packet_data


    def process_batch(self, segments_batch, segment_ids_batch, embedding_packet_data, embedding_packet_size):
        # pad the batch of tokens and re-create attention mask
        input_ids = pad_sequence(segments_batch, batch_first=True, padding_value=self.pad_token_id).to("cuda")
        target_position_mask = torch.isin(input_ids, self.target_token_ids).unsqueeze(2)
        if target_position_mask.any():
            attention_mask = (input_ids > self.pad_token_id).int()
            # get embeddings of the batch
            with torch.no_grad():
                outputs = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            embeddings = outputs.last_hidden_state
            target_indices = torch.unique(torch.where(target_position_mask, embeddings, 0).nonzero(as_tuple=False)[:,:2], dim=0)
            embeddings_batch = {}
            this_batch_counter = 0
            for idx in target_indices:
                segment_id = idx[0]
                token_id = input_ids[segment_id, idx[1]]
                token_id_item = token_id.item()
                lemma = self.id2lemma[token_id_item]
                if self.target_counter[lemma] < 1000:
                    if embeddings_batch.get(lemma) is None:
                        embeddings_batch[lemma] = [[],[]]
                    embedding = embeddings[segment_id, idx[1]]
                    embeddings_batch[lemma][0].append(embedding.unsqueeze(0).detach().cpu())   
                    embeddings_batch[lemma][1].append(segment_ids_batch[segment_id])
                    this_batch_counter += 1
                    self.target_counter[lemma] += 1
                    if self.target_counter[lemma] == 1000:
                        self.target_token_ids = self.target_token_ids[
                            ~torch.isin(self.target_token_ids, torch.tensor(self.lemma2id[lemma]).to("cuda"))
                        ]
            embedding_packet_data = self._process_embedding_packet_data(embeddings_batch, embedding_packet_data)
            embedding_packet_size += self.embedding_size * this_batch_counter # ignoring the size of containers (dict and lists)
            if embedding_packet_size > self.max_packet_size:
                embedding_packet_data, embedding_packet_size = self.save_embeddings_packet(embedding_packet_data)
        return embedding_packet_data, embedding_packet_size