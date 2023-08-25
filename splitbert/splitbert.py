import pandas as pd
import numpy as np
import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy

from tqdm import tqdm
from torch.utils.data import TensorDataset
from transformers import BertModel

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import modeling_outputs
from typing import Optional, Tuple, Union, Any
from sklearn.metrics import f1_score

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # print(sum_mask)
    return sum_embeddings / sum_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print(pe.shape)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # print(pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_batch_embeddings(model, fc_layer, fc_layer_for_pc, embedding_size, input_ids, attention_mask, sentence_count,
                         max_sentences, batch_embeddings, is_all, device, output_attentions=False):
    if is_all:
        max_sentences = 1

    for i in range(len(input_ids)):
        bert_outputs = torch.empty(size=(max_sentences, 1, embedding_size)).to(device)
        # print(bert_outputs.shape) 1, 1, 384

        model_output = model(input_ids[i][0:sentence_count[i]],
                             attention_mask=attention_mask[i][0:sentence_count[i]],
                             output_attentions=True)

        if output_attentions:
            attention = model_output.attentions
        # for a in model_output.attentions:
        #     print(a.shape)
        # exit()

        if is_all:
            model_output = model_output.pooler_output
            model_output = fc_layer_for_pc(model_output)

        else:
            model_output = mean_pooling(model_output, attention_mask[i][0:sentence_count[i]])

        for j in range(sentence_count[i].item()):
            bert_outputs[j] = model_output[j]

        for j in range(sentence_count[i].item(), max_sentences):
            bert_outputs[j] = torch.zeros(1, embedding_size).to(device)

        # nn.Linear
        # bert_outputs_fc = fc_layer(bert_outputs)
        # bert_outputs_fc = bert_outputs_fc.swapaxes(0, 1)
        # batch_embeddings[i] = bert_outputs_fc
        batch_embeddings[i] = bert_outputs.swapaxes(0, 1)

    if output_attentions:
        return batch_embeddings, attention
    return batch_embeddings


def make_masks(max_sentences, count, device, max_post_len=0, max_comment_len=0, mode='sep'):
    if mode == 'sep':
        mask = (torch.triu(torch.ones(max_sentences, max_sentences)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)

        zeros = torch.zeros(1, count.item()).to(device)
        ones = torch.ones(1, max_sentences - count.item()).to(device)
        key_padding_mask = torch.cat([zeros, ones], dim=1).type(torch.bool)

    else:
        max_len = max_post_len + max_comment_len
        mask = (torch.triu(torch.ones(max_len, max_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)

        if mode == 'concat_seq':
            key_padding_mask = torch.ones(1, max_len).to(device)

            i = 0
            while i < count[0].item():
                key_padding_mask[0][i] = 0
                i += 1

            i += max_post_len - count[0].item()

            while i < max_post_len + count[1].item():
                key_padding_mask[0][i] = 0
                i += 1

        else:  # concat_all
            zeros = torch.zeros(1, count[0].item() + count[1].item()).to(device)
            ones = torch.ones(1, max_len - (count[0].item() + count[1].item())).to(device)
            key_padding_mask = torch.cat([zeros, ones], dim=1)

        key_padding_mask = key_padding_mask.type(torch.bool)

    return mask, key_padding_mask


def normalize_tensor(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


class SplitBertTransformerModel(nn.Module):

    def __init__(self, num_labels, embedding_size, max_sentences, max_len1, max_len2, device):
        super().__init__()
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.max_sentences = max_sentences
        self.max_len1 = max_len1
        self.max_len2 = max_len2
        self.device = device

        self.sbert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.fc_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding_size, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.pe = PositionalEncoding(self.embedding_size, max_len=self.max_sentences)
        self.multihead_attn = nn.MultiheadAttention(self.embedding_size, 12)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier1 = nn.Linear(self.embedding_size, self.embedding_size//2)
        self.classifier2 = nn.Linear(self.embedding_size//2, self.num_labels)
        self.classifier3 = nn.Linear(self.embedding_size, self.num_labels)
        # self.post_init()

    def forward(
            self,
            labels: Optional[torch.tensor] = None,
            input_ids1: Optional[torch.tensor] = None,
            input_ids2: Optional[torch.tensor] = None,
            attention_mask1: Optional[torch.tensor] = None,
            attention_mask2: Optional[torch.tensor] = None,
            sentence_count1: Optional[torch.tensor] = None,
            sentence_count2: Optional[torch.tensor] = None,
            mode: Optional[torch.tensor] = None):

        current_batch_size = len(input_ids1)

        batch_encoder_embeddings = torch.empty(size=(current_batch_size, 1, self.max_sentences, self.embedding_size),
                                               requires_grad=True).to(self.device)
        batch_decoder_embeddings = torch.empty(size=(current_batch_size, 1, self.max_sentences, self.embedding_size),
                                               requires_grad=True).to(self.device)

        batch_encoder_embeddings = get_batch_embeddings(self.sbert, self.fc_layer, None, self.embedding_size, input_ids1,
                                                        attention_mask1, sentence_count1, self.max_sentences,
                                                        batch_encoder_embeddings, False, self.device)
        batch_decoder_embeddings = get_batch_embeddings(self.sbert, self.fc_layer, None, self.embedding_size, input_ids2,
                                                        attention_mask2, sentence_count2, self.max_sentences,
                                                        batch_decoder_embeddings, False, self.device)

        outputs = torch.empty(size=(current_batch_size, self.embedding_size), requires_grad=True).to(self.device)
        attentions = {'encoder_attentions': [], 'decoder_attentions': [], 'cross_attentions': []}

        for encoder_embeddings, decoder_embeddings, count1, count2, i in zip(batch_encoder_embeddings,
                                                                             batch_decoder_embeddings,
                                                                             sentence_count1,
                                                                             sentence_count2,
                                                                             range(current_batch_size)):
            encoder_embeddings = encoder_embeddings.swapaxes(0, 1)
            decoder_embeddings = decoder_embeddings.swapaxes(0, 1)

            # add positional encoding
            encoder_embeddings = self.pe(encoder_embeddings)

            # make masks
            src_mask, src_key_padding_mask = make_masks(self.max_sentences, count1, self.device)
            tgt_mask, tgt_key_padding_mask = make_masks(self.max_sentences, count2, self.device)

            encoder_output = self.encoder(encoder_embeddings,
                                          mask=src_mask,
                                          src_key_padding_mask=src_key_padding_mask)

            # shape: (len, 1, 384)
            now_attention = torch.empty(size=(count1, 1, self.embedding_size), requires_grad=True).to(self.device)
            encoder_attentions = encoder_output[:count1]

            for now in range(len(encoder_attentions)):
                now_attention[now] = normalize_tensor(encoder_attentions[now])

            attention = self.encoder_layer.self_attn(encoder_attentions, encoder_attentions, encoder_attentions)
            attentions['encoder_attentions'].append(attention[1].unsqueeze(0))

            decoder_output = self.decoder(tgt=decoder_embeddings, memory=encoder_output,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask)

            now_attention = torch.empty(size=(count2, 1, self.embedding_size), requires_grad=True).to(self.device)
            decoder_attentions = decoder_output[:count2]
            for now in range(len(decoder_attentions)):
                now_attention[now] = normalize_tensor(decoder_attentions[now])

            attention = self.decoder_layer.self_attn(decoder_attentions, decoder_attentions, decoder_attentions)
            attentions['decoder_attentions'].append(attention[1].unsqueeze(0))

            decoder_embedding_attention = decoder_embeddings[:count2]
            attention = self.decoder_layer.multihead_attn(decoder_embedding_attention, encoder_attentions,
                                                          encoder_attentions)
            attentions['cross_attentions'].append(attention[1].unsqueeze(0))

            attn_mask = copy.deepcopy(src_key_padding_mask.to(self.device))

            for j in range(1, self.max_sentences):
                if j < count2:
                    attn_mask = torch.cat([attn_mask, src_key_padding_mask], dim=0)
                else:
                    attn_mask = torch.cat([attn_mask, torch.tensor([True]*self.max_sentences).unsqueeze(0).to(self.device)], dim=0)

            decoder_output = torch.mean(decoder_output[:count2], dim=0).squeeze(0)
            outputs[i] = decoder_output

        outputs = self.classifier1(outputs)
        logits = self.classifier2(outputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        # loss = loss_fct(logits, labels)

        return (loss, logits, outputs, attentions)


class SplitBertEncoderModel(nn.Module):

    def __init__(self, num_labels, embedding_size, max_len):
        super().__init__()
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.max_len = max_len

        self.sbert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.fc_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.pe = PositionalEncoding(self.embedding_size, max_len=self.max_len)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier1 = nn.Linear(self.embedding_size, self.embedding_size//2)
        self.classifier2 = nn.Linear(self.embedding_size//2, self.num_labels)
        self.classifier3 = nn.Linear(self.embedding_size, self.num_labels)
        # self.post_init()

    def forward(
            self,
            labels: Optional[torch.tensor] = None,
            input_ids: Optional[torch.tensor] = None,
            attention_mask: Optional[torch.tensor] = None,
            sentence_count: Optional[torch.tensor] = None,
            indexes: Optional[torch.tensor] = None
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.SequenceClassifierOutput]:

        current_batch_size = len(input_ids)
        batch_embeddings = torch.empty(size=(current_batch_size, 1, self.max_len, self.embedding_size),
                                       requires_grad=True).to('cuda')

        batch_embeddings = get_batch_embeddings(self.sbert, self.fc_layer, self.embedding_size, input_ids,
                                                attention_mask, sentence_count, self.max_len, batch_embeddings)

        encoder_outputs = torch.empty(size=(current_batch_size, self.embedding_size), requires_grad=True).to('cuda')

        for embeddings, count, i in zip(batch_embeddings, sentence_count, range(current_batch_size)):
            embeddings = embeddings.swapaxes(0, 1)

            # add positional encoding
            embeddings = self.pe(embeddings)

            # make masks
            src_mask, src_key_padding_mask = make_masks(self.max_len, count)

            encoder_output = self.encoder(embeddings, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

            # mean
            encoder_output = torch.mean(encoder_output[:count], dim=0).squeeze(0)

            # last output
            # encoder_output = encoder_output[count-1].squeeze(0)

            encoder_outputs[i] = encoder_output

        encoder_outputs = self.classifier1(encoder_outputs)
        logits = self.classifier2(encoder_outputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs
        )


class SplitBertConcatEncoderModel(nn.Module):
    def __init__(self, num_labels, embedding_size, max_len, max_post_len, max_comment_len, device, target, concat_mode,
                 attention_mode=False, output_attentions=False):
        super().__init__()
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.max_post_len = max_post_len
        self.max_comment_len = max_comment_len
        self.target = target
        self.device = device
        self.concat_mode = concat_mode
        self.attention_mode=attention_mode
        self.output_attentions = output_attentions

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.sbert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.fc_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_layer_for_pc = nn.Linear(768, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.pe = PositionalEncoding(self.embedding_size, max_len=self.max_len)
        self.layer_norm = nn.LayerNorm(self.embedding_size)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.classifier1 = nn.Linear(self.embedding_size * (2 if target == 'post_comment' else 3), self.embedding_size)
        self.classifier2 = nn.Linear(self.embedding_size, self.num_labels)
        # self.attention_fc_layer = nn.Linear(self)
        # self.classifier3 = nn.Linear(self.embedding_size, self.num_labels)
        # self.post_init()

    def forward(
            self,
            labels: Optional[torch.tensor] = None,
            input_ids1: Optional[torch.tensor] = None,
            input_ids2: Optional[torch.tensor] = None,
            input_ids3: Optional[torch.tensor] = None,
            attention_mask1: Optional[torch.tensor] = None,
            attention_mask2: Optional[torch.tensor] = None,
            attention_mask3: Optional[torch.tensor] = None,
            sentence_count1: Optional[torch.tensor] = None,
            sentence_count2: Optional[torch.tensor] = None,
            sentence_count3: Optional[torch.tensor] = None,
            mode: Optional[torch.tensor] = None
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.SequenceClassifierOutput]:

        if input_ids2 is None:
            input_ids_list = [input_ids1]
            attention_mask_list = [attention_mask1]
            sentence_count_list = [sentence_count1]
        elif input_ids3 is None:
            input_ids_list = [input_ids1, input_ids2]
            attention_mask_list = [attention_mask1, attention_mask2]
            sentence_count_list = [sentence_count1, sentence_count2]
        else:
            input_ids_list = [input_ids1, input_ids2, input_ids3]
            attention_mask_list = [attention_mask1, attention_mask2, attention_mask3]
            sentence_count_list = [sentence_count1, sentence_count2, sentence_count3]

        current_batch_size = len(input_ids1)

        batch_embeddings_list = []
        encoder_outputs_list = []
        attention_list = []
        attentions = []

        now = 0

        if self.concat_mode == 'sep':
            for input_ids, attention_mask, sentence_count, now_mode in zip(input_ids_list, attention_mask_list,
                                                                           sentence_count_list, mode):
                if now_mode == 'all':
                    model = self.bert
                    is_all = True
                    max_sentence = 1
                else:
                    model = self.sbert
                    is_all = False
                    max_sentence = self.max_len

                batch_embeddings = torch.empty(size=(current_batch_size, 1, max_sentence, self.embedding_size),
                                               requires_grad=True).to(self.device)

                if self.output_attentions:
                    batch_embeddings, attention = get_batch_embeddings(model, self.fc_layer, self.fc_layer_for_pc,
                                                                       self.embedding_size, input_ids, attention_mask,
                                                                       sentence_count, self.max_len, batch_embeddings,
                                                                       is_all, self.device, True)
                else:
                    batch_embeddings = get_batch_embeddings(model, self.fc_layer, self.fc_layer_for_pc,
                                                            self.embedding_size, input_ids, attention_mask,
                                                            sentence_count, self.max_len, batch_embeddings, is_all,
                                                            self.device)
                    batch_embeddings_list.append(batch_embeddings)

                encoder_outputs = torch.empty(size=(current_batch_size, self.embedding_size), requires_grad=True).to(self.device)
                attention_outputs = torch.empty(size=(current_batch_size, self.max_len + 4, self.embedding_size), requires_grad=True).to(self.device)

                for embeddings, count, i in zip(batch_embeddings, sentence_count, range(current_batch_size)):
                    embeddings = embeddings.swapaxes(0, 1)


                    if is_all:  # no zero padding -> no src mask
                        encoder_output = self.encoder(embeddings)
                    else:
                        # add positional encoding
                        embeddings = self.pe(embeddings)

                        # make masks
                        src_mask, src_key_padding_mask = make_masks(self.max_len, count, self.device)

                        encoder_output = self.encoder(embeddings, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
                        # attention_output = self.encoder(embeddings[:count])

                        if len(encoder_outputs_list) == current_batch_size:
                            encoder_outputs_list[i] = torch.cat([encoder_outputs_list[i], encoder_output[:count]],
                                                                dim=0)
                        else:
                            encoder_outputs_list.append(encoder_output[:count])

            for i in range(current_batch_size):
                # attentions = self.encoder_layer.self_attn(encoder_outputs_list[i], encoder_outputs_list[i],
                #                                           encoder_outputs_list[i], attn_mask=src_mask,
                #                                           key_padding_mask=src_key_padding_mask)
                attentions = self.encoder_layer.self_attn(encoder_outputs_list[i], encoder_outputs_list[i],
                                                          encoder_outputs_list[i])

                attention = attentions[0]

                # mean
                if self.attention_mode:
                    if self.attention_mode == 'attention_residual':
                        # add & norm
                        residual = encoder_outputs_list[i]
                        attention = self.dropout(attention) + residual
                        attention = self.layer_norm(attention)
                    encoder_output = torch.mean(attention, dim=0).squeeze(0)
                else:
                    encoder_output = torch.mean(encoder_outputs_list[i], dim=0).squeeze(0)


                # last output
                # encoder_output = encoder_output[count-1].squeeze(0)

                encoder_outputs[i] = encoder_output
                #     if now == 0:
                #         result_outputs = encoder_outputs
                #     else:
                #         result_outputs = torch.cat([result_outputs, encoder_outputs], dim=1)
                #
                #     now += 1

            '''
            for i in range(len(attentions)):
                attentions[i] = torch.cat(attentions[i], dim=0)
                attentions[i] = self.encoder_layer.self_attn(attentions[i], attentions[i], attentions[i])[1]
            '''


        elif self.concat_mode == 'concat_seq':
            for input_ids, attention_mask, sentence_count, now_mode, now_max_len in zip(input_ids_list,
                                                                                        attention_mask_list,
                                                                                        sentence_count_list, mode,
                                                                                        (self.max_post_len,
                                                                                         self.max_comment_len)):

                batch_embeddings = torch.empty(size=(current_batch_size, 1, now_max_len, self.embedding_size),
                                               requires_grad=True).to(self.device)

                if now_mode == 'all':
                    model = self.bert
                    is_all = True
                else:
                    model = self.sbert
                    is_all = False

                if self.output_attentions:
                    batch_embeddings, attention = get_batch_embeddings(model, self.fc_layer, self.fc_layer_for_pc,
                                                                       self.embedding_size, input_ids, attention_mask,
                                                                       sentence_count, now_max_len, batch_embeddings,
                                                                       is_all, self.device, True)
                else:
                    batch_embeddings = get_batch_embeddings(model, self.fc_layer, self.fc_layer_for_pc,
                                                            self.embedding_size, input_ids, attention_mask,
                                                            sentence_count, now_max_len, batch_embeddings, is_all,
                                                            self.device)
                batch_embeddings_list.append(batch_embeddings)

            batch_embeddings = torch.cat([batch_embeddings_list[0], batch_embeddings_list[1]], dim=2)

            encoder_outputs = torch.empty(size=(current_batch_size, self.embedding_size), requires_grad=True).to(
                self.device)

            i = 0
            for embeddings, p_count, c_count in zip(batch_embeddings, sentence_count_list[0], sentence_count_list[1]):
                embeddings = embeddings.swapaxes(0, 1)
                src_mask, src_key_padding_mask = make_masks(self.max_post_len, [p_count, c_count], self.device,
                                                             self.max_post_len, self.max_comment_len, self.concat_mode)

                encoder_output = self.encoder(embeddings, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
                encoder_outputs_list.append(encoder_output.swapaxes(0, 1))

                indices_to_mean = list(range(p_count)) + list(range(self.max_post_len, self.max_post_len + c_count))
                encoder_output = torch.mean(encoder_output[indices_to_mean], dim=0).squeeze(0)
                encoder_outputs[i] = encoder_output
                i += 1

        else:  # concat_all
            for input_ids, attention_mask, sentence_count, now_mode, now_max_len in zip(input_ids_list,
                                                                                        attention_mask_list,
                                                                                        sentence_count_list, mode,
                                                                                        (self.max_post_len,
                                                                                         self.max_comment_len)):

                batch_embeddings = torch.empty(size=(current_batch_size, 1, now_max_len, self.embedding_size),
                                               requires_grad=True).to(self.device)

                if now_mode == 'all':
                    model = self.bert
                    is_all = True
                else:
                    model = self.sbert
                    is_all = False

                if self.output_attentions:
                    batch_embeddings, attention = get_batch_embeddings(model, self.fc_layer, self.fc_layer_for_pc,
                                                                       self.embedding_size, input_ids, attention_mask,
                                                                       sentence_count, now_max_len, batch_embeddings,
                                                                       is_all, self.device, True)
                else:
                    batch_embeddings = get_batch_embeddings(model, self.fc_layer, self.fc_layer_for_pc,
                                                            self.embedding_size, input_ids, attention_mask,
                                                            sentence_count, now_max_len, batch_embeddings, is_all,
                                                            self.device)
                batch_embeddings_list.append(batch_embeddings)

            batch_embeddings = torch.cat([batch_embeddings_list[0], batch_embeddings_list[1]], dim=2)

            for i in range(len(batch_embeddings)):
                non_zero_rows = batch_embeddings[i][0][batch_embeddings[i][0].sum(dim=1) != 0]
                zero_rows = torch.zeros((batch_embeddings[i][0].shape[0] - non_zero_rows.shape[0], self.embedding_size),
                                        dtype=torch.int, device=self.device)
                batch_embeddings[i] = torch.cat([non_zero_rows, zero_rows])

            encoder_outputs = torch.empty(size=(current_batch_size, self.embedding_size), requires_grad=True).to(
                self.device)

            i = 0
            for embeddings, p_count, c_count in zip(batch_embeddings, sentence_count_list[0], sentence_count_list[1]):
                embeddings = embeddings.swapaxes(0, 1)
                src_mask, src_key_padding_mask = make_masks(self.max_post_len, [p_count, c_count], self.device,
                                                             self.max_post_len, self.max_comment_len, self.concat_mode)

                encoder_output = self.encoder(embeddings, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

                now_attention = torch.empty(size=(p_count+c_count, 1, self.embedding_size), requires_grad=True).to(self.device)
                for now in range(len(now_attention)):
                    now_attention[now] = normalize_tensor(encoder_output[now])

                attention = self.encoder_layer.self_attn(now_attention, now_attention, now_attention)[1]
                # attention = self.encoder_layer.self_attn(encoder_output, encoder_output, encoder_output,
                #                                          attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[1]
                attentions[i].append(attention)
                encoder_outputs_list.append(encoder_output.swapaxes(0, 1))

                encoder_output = torch.mean(encoder_output[:(p_count+c_count)], dim=0).squeeze(0)
                encoder_outputs[i] = encoder_output
                i += 1

        # if self.target != 'reply' and self.concat_mode == 'sep':
        #     result_outputs = self.classifier1(result_outputs)

        # if self.concat_mode == 'sep':
        #     logits = self.classifier2(result_outputs)
        # else:
        #     logits = self.classifier2(encoder_outputs)
        logits = self.classifier2(encoder_outputs)

        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.squeeze(), labels.squeeze())
        # loss = loss_fct(logits, labels)

        if self.output_attentions:
            return modeling_outputs.SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=encoder_outputs_list,
                attentions=attentions
            )

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs_list
        )


def conduct_input_ids_and_attention_masks(tokenizer, str_values, label_values, score_values, index_values,
                                          max_count, target):
    tensor_datasets = []
    total_input_ids = []
    total_attention_masks = []
    total_sentence_count = []

    for value in str_values:
        input_ids_list = []
        attention_masks_list = []
        sentence_count_list = []
        for contents in value:
            result = tokenizer(contents,
                               pad_to_max_length=True, truncation=True, max_length=256, return_tensors='pt')

            input_ids = result['input_ids']
            attention_masks = result['attention_mask']

            sentence_count_list.append(len(input_ids))

            # add zero pads to make all tensors' dimension (max_sentences, 128)
            pad = (0, 0, 0, max_count-len(input_ids))
            input_ids = nn.functional.pad(input_ids, pad, "constant", 0)  # effectively zero padding
            attention_masks = nn.functional.pad(attention_masks, pad, "constant", 0)

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_masks)

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_masks = torch.stack(attention_masks_list, dim=0)
        sentence_counts = torch.tensor(sentence_count_list)

        total_input_ids.append(input_ids)
        total_attention_masks.append(attention_masks)
        total_sentence_count.append(sentence_counts)

        print(input_ids.shape, attention_masks.shape, sentence_counts.shape)
    print(total_sentence_count)

    labels = torch.tensor(label_values.astype(int))
    scores = torch.tensor(score_values.astype(float))
    indexes = torch.tensor(index_values.astype(int))

    # 0: posts / 1: comments
    if target == 'post_comment':
        return TensorDataset(labels, total_input_ids[0], total_input_ids[1],
                             total_attention_masks[0], total_attention_masks[1],
                             total_sentence_count[0], total_sentence_count[1], scores, indexes)
    elif target == 'reply':
        return TensorDataset(labels, total_input_ids[0], total_attention_masks[0], total_sentence_count[0],
                             scores, indexes)
    # 0: posts / 1: comments / 2: reply
    else:  # triple
        return TensorDataset(labels, total_input_ids[0], total_input_ids[1], total_input_ids[2],
                             total_attention_masks[0], total_attention_masks[1], total_attention_masks[2],
                             total_sentence_count[0], total_sentence_count[1], total_sentence_count[2], scores, indexes)


def conduct_input_ids_and_attention_masks_is_es(tokenizer, str_values, label_values, index_values, max_count, target):
    tensor_datasets = []
    total_input_ids = []
    total_attention_masks = []
    total_sentence_count = []

    for value in str_values:
        input_ids_list = []
        attention_masks_list = []
        sentence_count_list = []
        for contents in value:
            result = tokenizer(contents,
                               pad_to_max_length=True, truncation=True, max_length=256, return_tensors='pt')

            input_ids = result['input_ids']
            attention_masks = result['attention_mask']

            sentence_count_list.append(len(input_ids))

            # add zero pads to make all tensors' dimension (max_sentences, 128)
            pad = (0, 0, 0, max_count-len(input_ids))
            input_ids = nn.functional.pad(input_ids, pad, "constant", 0)  # effectively zero padding
            attention_masks = nn.functional.pad(attention_masks, pad, "constant", 0)

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_masks)

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_masks = torch.stack(attention_masks_list, dim=0)
        sentence_counts = torch.tensor(sentence_count_list)

        total_input_ids.append(input_ids)
        total_attention_masks.append(attention_masks)
        total_sentence_count.append(sentence_counts)

        print(input_ids.shape, attention_masks.shape, sentence_counts.shape)
    print(total_sentence_count)

    labels = torch.tensor(label_values.astype(int))
    indexes = torch.tensor(index_values.astype(int))

    # 0: posts / 1: comments
    if target == 'post_comment':
        return TensorDataset(labels, total_input_ids[0], total_input_ids[1],
                             total_attention_masks[0], total_attention_masks[1],
                             total_sentence_count[0], total_sentence_count[1], indexes)
    elif target == 'reply':
        return TensorDataset(labels, total_input_ids[0], total_attention_masks[0], total_sentence_count[0], indexes)
    # 0: posts / 1: comments / 2: reply
    else:  # triple
        return TensorDataset(labels, total_input_ids[0], total_input_ids[1], total_input_ids[2],
                             total_attention_masks[0], total_attention_masks[1], total_attention_masks[2],
                             total_sentence_count[0], total_sentence_count[1], total_sentence_count[2], indexes)


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='macro'), f1_score(labels_flat, preds_flat, average='micro')


def accuracy_per_class(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class {label} : ', end='')
        print(f'{len(y_preds[y_preds == label])}/{len(y_true)}')


def evaluate(dataloader_val, model, device, target, labels, mode):
    model.eval()
    loss_val_total = 0

    embeddings, predictions, true_vals, true_scores, indexes = [], [], [], [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        one_hot_labels = torch.nn.functional.one_hot(batch[0], num_classes=len(labels))

        if target == 'post_comment':
            inputs = {'labels': one_hot_labels.type(torch.float),
                      'input_ids1': batch[1],
                      'input_ids2': batch[2],
                      'attention_mask1': batch[3],
                      'attention_mask2': batch[4],
                      'sentence_count1': batch[5],
                      'sentence_count2': batch[6],
                      'mode': mode
                      }
        elif target == 'reply':
            inputs = {'labels': one_hot_labels.type(torch.float),
                      'input_ids1': batch[1],
                      'attention_mask1': batch[2],
                      'sentence_count1': batch[3],
                      'mode': mode
                      }
        else:
            inputs = {'labels': one_hot_labels.type(torch.float),
                      'input_ids1': batch[1],
                      'input_ids2': batch[2],
                      'input_ids3': batch[3],
                      'attention_mask1': batch[4],
                      'attention_mask2': batch[5],
                      'attention_mask3': batch[6],
                      'sentence_count1': batch[7],
                      'sentence_count2': batch[8],
                      'sentence_count3': batch[9],
                      'mode': mode
                      }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        hidden_states = outputs[2]
        loss_val_total += loss.item()

        # hidden_states = hidden_states.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        label_ids = batch[0].cpu().numpy()

        if target == 'post_comment':
            score_ids = batch[7].cpu().numpy()
            index_ids = batch[8].cpu().numpy()
        elif target == 'reply':
            score_ids = batch[4].cpu().numpy()
            index_ids = batch[5].cpu().numpy()
        else:
            score_ids = batch[10].cpu().numpy()
            index_ids = batch[11].cpu().numpy()

        # embeddings.append(hidden_states)
        predictions.append(logits)
        true_vals.append(label_ids)
        true_scores.append(score_ids)
        indexes.append(index_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    # embeddings = np.concatenate(embeddings, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    true_scores = np.concatenate(true_scores, axis=0)
    indexes = np.concatenate(indexes, axis=0)

    accuracy_per_class(predictions, true_vals)

    return loss_val_avg, predictions, true_vals, true_scores, indexes


def train(model, device, dataset_train, dataset_val, labels, target, path, mode):
    if target == 'post_comment':
        mode_path = f"{mode[0]}_{mode[1]}"
    elif target == 'reply':
        mode_path = mode[0]
    else:
        mode_path = f"{mode[0]}_{mode[1]}_{mode[2]}"

    create_folder(path + f'/csv/splitbert_classifier/{target}/{mode_path}')
    create_folder(path + f'/splitbert/model/{mode_path}/')

    optimizer = AdamW(model.parameters(),
                      lr=2e-4,
                      eps=1e-8)
    epochs = 10
    batch_size = 4

    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)
    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    model.to(device)

    training_result = []

    # Training Loop
    for epoch in tqdm(range(1, epochs + 1)):
        result_for_tsne = []
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        i = 0
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            one_hot_labels = torch.nn.functional.one_hot(batch[0], num_classes=len(labels))

            if target == 'post_comment':
                inputs = {'labels': one_hot_labels.type(torch.float),
                          'input_ids1': batch[1],
                          'input_ids2': batch[2],
                          'attention_mask1': batch[3],
                          'attention_mask2': batch[4],
                          'sentence_count1': batch[5],
                          'sentence_count2': batch[6],
                          'mode': mode
                          }
            elif target == 'reply':
                inputs = {'labels': one_hot_labels.type(torch.float),
                          'input_ids1': batch[1],
                          'attention_mask1': batch[2],
                          'sentence_count1': batch[3],
                          'mode': mode
                          }
            else:
                inputs = {'labels': one_hot_labels.type(torch.float),
                          'input_ids1': batch[1],
                          'input_ids2': batch[2],
                          'input_ids3': batch[3],
                          'attention_mask1': batch[4],
                          'attention_mask2': batch[5],
                          'attention_mask3': batch[6],
                          'sentence_count1': batch[7],
                          'sentence_count2': batch[8],
                          'sentence_count3': batch[9],
                          'mode': mode
                          }

            # check parameters are training
            '''
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(name, param.requires_grad)
            '''

            outputs = model(**inputs)
            loss = outputs[0]
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            predictions = np.concatenate([logits], axis=0)

            # 총 loss 계산.
            loss_train_total += loss.item()
            loss.backward()
            # print(loss.requires_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # gradient를 이용해 weight update.
            optimizer.step()
            # scheduler를 이용해 learning rate 조절.
            scheduler.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
            i += 1
        torch.save(model.state_dict(),
                   f'{path}/splitbert/model/{mode_path}/epoch_{epoch}.model')
        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        val_loss, predictions, true_vals, true_scores, indexes = evaluate(dataloader_validation, model,
                                                                                      device, target, labels, mode)
        preds_flat = np.argmax(predictions, axis=1).flatten()
        labels_flat = true_vals.flatten()
        scores_flat = true_scores.flatten()
        indexes_flat = indexes.flatten()
        # print(type(embeddings))

        val_f1_macro, val_f1_micro = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Macro, Micro): {val_f1_macro}, {val_f1_micro}')
        training_result.append([epoch, loss_train_avg, val_loss, val_f1_macro, val_f1_micro])

        tsne_df = pd.DataFrame({'prediction': preds_flat, 'label': labels_flat,
                                'score': scores_flat, 'index': indexes_flat})
        tsne_df.to_csv(path + f'/csv/splitbert_classifier/{target}/{mode_path}/epoch_{epoch}_result.csv')

    fields = ['epoch', 'training_loss', 'validation_loss', 'f1_score_macro', 'f1_score_micro']

    with open(
            path + f'/csv/splitbert_classifier/{target}/{mode_path}/training_result.csv',
            'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(training_result)