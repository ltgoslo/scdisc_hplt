from collections import defaultdict
import json
import gzip
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence


class Embedder:
    def __init__(self, embeddings_dir, batch_size, language, max_packet_size, model_name=None):
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
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, use_safetensors=False)
        full_model.eval()
        full_model.to('cuda') # GPU
        self.encoder_model = full_model.get_encoder()
        print(f"Model loaded: {model_name}", flush=True)
        # Load target words
        with open(os.path.join("../languages", language, 'target_words.json')) as f:
            target_words = json.load(f)
        self.target_token_ids = torch.tensor([token_id for token_id in target_words.values()]).to("cuda")
        self.target_counter = defaultdict(int)


    def save_embeddings_packet(self, embedding_packet_data):
        self.embedding_packet_count += 1
        output_file = os.path.join(self.embeddings_dir, str(self.embedding_packet_count) + 'pt.gz')
        with gzip.GzipFile(output_file, 'wb') as f:
            torch.save(embedding_packet_data, f)
        return {token_id.item(): [[], []] for token_id in self.target_token_ids}, 0


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
            embeddings_batch = {token_id.item(): [[], []] for token_id in self.target_token_ids}
            this_batch_counter = 0
            for idx in target_indices:
                segment_id = idx[0]
                token_id = input_ids[segment_id, idx[1]]
                token_id_item = token_id.item()
                if self.target_counter[token_id_item] < 999:
                    embedding = embeddings[segment_id, idx[1]]
                    embeddings_batch[token_id_item][0].append(embedding.detach().cpu())   
                    embeddings_batch[token_id_item][1].append(segment_ids_batch[segment_id])
                    this_batch_counter += 1
                    self.target_counter[token_id_item] += 1
                    if self.target_counter[token_id_item] == 999:
                        self.target_token_ids = self.target_token_ids[self.target_token_ids != token_id_item]

            for token_id in embeddings_batch.keys():
                embedding_packet_data[token_id][0].extend(embeddings_batch[token_id][0])
                embedding_packet_data[token_id][1].extend(embeddings_batch[token_id][1])

            embedding_packet_size += self.embedding_size * this_batch_counter # ignoring the size of containers (dict and lists)
            if embedding_packet_size > self.max_packet_size:
                embedding_packet_data, embedding_packet_size = self.save_embeddings_packet(embedding_packet_data)
        return embedding_packet_data, embedding_packet_size