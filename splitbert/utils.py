import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset


def conduct_input_ids_and_attention_masks(tokenizer, str_values, label_values, score_values, index_values,
                                          max_count, target):
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


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
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


def get_batch_embeddings(model, fc_layer_for_pc, embedding_size, input_ids, attention_mask, sentence_count,
                         max_sentences, batch_embeddings, is_all, device, output_attentions=False):
    if is_all:
        max_sentences = 1

    for i in range(len(input_ids)):
        bert_outputs = torch.empty(size=(max_sentences, 1, embedding_size)).to(device)

        model_output = model(input_ids[i][0:sentence_count[i]],
                             attention_mask=attention_mask[i][0:sentence_count[i]],
                             output_attentions=True)

        if output_attentions:
            attention = model_output.attentions

        if is_all:
            model_output = model_output.pooler_output
            model_output = fc_layer_for_pc(model_output)

        else:
            model_output = mean_pooling(model_output, attention_mask[i][0:sentence_count[i]])

        count = min(sentence_count[i].item(), max_sentences)

        for j in range(count):
            bert_outputs[j] = model_output[j]

        for j in range(count, max_sentences):
            bert_outputs[j] = torch.zeros(1, embedding_size).to(device)

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
