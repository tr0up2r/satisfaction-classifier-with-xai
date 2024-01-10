import sys
sys.path.append('../splitbert/')

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients
from sklearn.model_selection import train_test_split
from spacy.lang.en import English
from paragraph_split import text_segmentation
from SplitBertEncoderAttentionModel import SplitBertEncoderAttentionModel
from utils import conduct_input_ids_and_attention_masks
from utils import make_masks
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import tqdm


def get_sequences(contents, mode):
    sequences = []
    nlp = English()
    nlp.add_pipe("sentencizer")

    if mode == 'all':
        for content in contents:
            sequences.append([content])
    elif mode == 'seg':
        for content in contents:
            sentences = list(map(lambda x: str(x), list(nlp(content).sents)))
            sequences.append(text_segmentation(sentences))
    else:  # sentences
        for content in contents:
            sequences.append(list(map(lambda x: str(x), list(nlp(content).sents))))

    return sequences


def normalize_tensor(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def forward_func_ig(inputs, p_count, c_count, model):
    embeddings = inputs
    encoder_outputs = torch.empty(size=(1, model.embedding_size * 2)).to(model.device)
    outputs_list = []

    non_zero_rows = embeddings[0][embeddings[0].sum(dim=1) != 0]
    zero_rows = torch.zeros((embeddings[0].shape[0] - non_zero_rows.shape[0], model.embedding_size),
                            dtype=torch.int, device=model.device)
    embeddings = torch.cat([non_zero_rows, zero_rows])
    embeddings = embeddings.unsqueeze(0)
    embeddings = embeddings.swapaxes(0, 1)
    outputs_list.append(embeddings)
    src_mask, src_key_padding_mask = make_masks(model.max_post_len, [p_count, c_count], model.device,
                                                model.max_post_len,
                                                model.max_comment_len, "concat_all")
    if model.encoder_mask_mode:
        encoder_output = model.encoder(embeddings, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    else:
        encoder_output = model.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
    encoder_outputs[0][:model.embedding_size] = torch.mean(encoder_output[:p_count + c_count], dim=0).squeeze(0)

    if model.attention_mask_mode:
        attention = model.mhead_attention(encoder_output, encoder_output, encoder_output, attn_mask=src_mask,
                                          key_padding_mask=src_key_padding_mask)[0]
    else:
        attention = \
        model.mhead_attention(encoder_output, encoder_output, encoder_output, key_padding_mask=src_key_padding_mask)[0]

    # mul mask - diagonal masking
    attention = attention.swapaxes(0, 2)
    mask = torch.tensor(
        [1] * (p_count + c_count) + [0] * (model.max_post_len + model.max_comment_len - (p_count + c_count))).to(
        model.device)
    attention = attention.mul(mask).swapaxes(0, 2)

    attention = torch.flatten(attention)
    attention = model.attn_classifier1(attention)
    attention = model.attn_classifier2(attention)
    encoder_outputs[0][model.embedding_size:] = attention

    encoder_outputs = model.mean_attn_layer(encoder_outputs)
    logits = model.classifier2(encoder_outputs)
    if model.softmax:
        logits = F.softmax(logits, dim=1)
    return logits


def prepare_model(labels, max_count, max_post, max_comment, path, epoch, encoder_mask_mode, attention_mask_mode, softmax):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model_path = f'{path}/epoch_{epoch}.model'

    model = SplitBertEncoderAttentionModel(num_labels=len(labels), embedding_size=384, max_len=max_count,
                                           max_post_len=max_post, max_comment_len=max_comment, device=device,
                                           target="post_comment", encoder_mask_mode=encoder_mask_mode,
                                           attention_mask_mode=attention_mask_mode, softmax=softmax)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to('cpu')
    model.eval()

    for param in model.sbert.parameters():
        param.requires_grad = False

    for param in model.bert.parameters():
        param.requires_grad = False

    return device, model, tokenizer


def construct_input_ref_pair(targets, tokenizer, max_count):
    input_ids_list, ref_input_ids_list, attention_masks_list, sentence_count_list = [], [], [], []

    for contents in targets:
        result = tokenizer(contents, pad_to_max_length=True, truncation=True, max_length=256, return_tensors='pt')

        input_ids = result['input_ids']
        sentence_count_list.append(torch.tensor(len(input_ids)).unsqueeze(0))
        attention_masks = result['attention_mask']

        pad = (0, 0, 0, max_count - len(input_ids))
        input_ids = nn.functional.pad(input_ids, pad, "constant", 0)
        attention_masks = nn.functional.pad(attention_masks, pad, "constant", 0)
        ref_input_ids = torch.zeros_like(input_ids)

        input_ids_list.append(input_ids.unsqueeze(0))
        ref_input_ids_list.append(ref_input_ids.unsqueeze(0))
        attention_masks_list.append(attention_masks.unsqueeze(0))

    return input_ids_list, ref_input_ids_list, attention_masks_list, sentence_count_list


def summarize_attributions(attribution):
    attributions = attribution.sum(dim=-1).squeeze(0)
    print(torch.norm(attributions))
    attributions = attributions / torch.norm(attributions)
    return attributions


def get_indexes(filename):
    index_df = pd.read_csv(filename, encoding='UTF-8')
    index_df.columns = ['Unnamed: 0', 'prediction', 'label', 'score', 'idx']
    val_index = sorted(list(index_df.idx.values))

    return val_index


def splitbert_integrated_gradient(index, labels, pc_model, device, tokenizer, max_count, post, comment,
                                  p_sentences, c_sentences, label, score, true_label, pred_label, target,
                                  visualize=False):
    def post_or_comment_or_reply(index):
        for i, sentences in enumerate(all_sentences):
            if all_tokens[index] in sentences:
                if i == 0:
                    return 'post'
                elif i == 1:
                    return 'comment'
                else:
                    return 'reply'

    ig = IntegratedGradients(forward_func_ig)
    input_ids, ref_input_ids, attention_masks, sentence_counts = construct_input_ref_pair([post, comment], tokenizer,
                                                                                          max_count)

    one_hot_labels = torch.nn.functional.one_hot(torch.tensor(label), num_classes=len(labels))
    inputs = {'labels': one_hot_labels.type(torch.float).to(device),
              'input_ids1': input_ids[0].to(device),
              'input_ids2': input_ids[1].to(device),
              'attention_mask1': attention_masks[0].to(device),
              'attention_mask2': attention_masks[1].to(device),
              'sentence_count1': sentence_counts[0].to(device),
              'sentence_count2': sentence_counts[1].to(device)
              }

    with torch.no_grad():
        inputs = pc_model(**inputs).hidden_states
    inputs = inputs[0]

    # inputs = torch.stack(embeddings, dim=0)
    baselines = torch.zeros((1, 14, 384))
    pred = forward_func_ig(inputs, sentence_counts[0], sentence_counts[1], pc_model)
    base_pred = forward_func_ig(baselines, sentence_counts[0], sentence_counts[1], pc_model)

    # pred = forward_func_ig(inputs, sentence_counts[0], sentence_counts[1], pc_model)
    # base_pred = forward_func_ig(baselines, sentence_counts[0], sentence_counts[1], pc_model)
    # print(f'pred: {pred}, base_pred: {base_pred}')

    result = []

    if target == 'pred':
        target_val = torch.argmax(pred)
    else:
        target_val = true_label

    if label == true_label and torch.argmax(pred) == pred_label:
        attribution, delta = ig.attribute(inputs=inputs, target=target_val,
                                          additional_forward_args=(sentence_counts[0], sentence_counts[1], pc_model),
                                          n_steps=50, internal_batch_size=1, return_convergence_delta=True)
        attributions = summarize_attributions(attribution)
        f_attributions = torch.flatten(attributions)
        f_attributions = f_attributions[f_attributions.nonzero()].squeeze(1)
        abs_attributions = list(map(abs, map(float, f_attributions)))
        idx_attributions = []
        for j in range(len(abs_attributions)):
            idx_attributions.append((j, abs_attributions[j], f_attributions[j].item()))
        idx_attributions.sort(key=lambda x: x[1], reverse=True)

        top3 = idx_attributions[:3]

        if visualize:
            all_sentences = [['[[post]]'], post, ['[[comment]]'], comment]
            all_tokens = [item for all_sentences in all_sentences for item in all_sentences]

            vis_attributions = []

            j = 0
            for i in range(len(all_tokens)):
                if all_tokens[i] in ['[[post]]', '[[comment]]']:
                    vis_attributions.append(0)
                else:
                    vis_attributions.append(f_attributions[j].item())
                    j += 1

            vis_attributions = torch.tensor(vis_attributions)

            score_vis = viz.VisualizationDataRecord(vis_attributions,
                                                    torch.max(torch.softmax(pred, dim=0)),
                                                    torch.argmax(pred),  # predicted label
                                                    f'{label}, {score}',  # true label
                                                    p_sentences + ' ' + c_sentences,
                                                    vis_attributions.sum(),
                                                    all_tokens,
                                                    delta)
            raw_text = ' '.join(post) + ' '.join(comment)

            print('\033[1m', 'Visualization For Score', '\033[0m')
            viz.visualize_text([score_vis])
            # print(f'pred: {pred}, base_pred: {base_pred}')
            # print(f'sub: {pred - base_pred}')
            print(vis_attributions)
            # print('delta: ', delta)

        else:
            where = []

            all_sentences = [post, comment]
            all_tokens = [item for all_sentences in all_sentences for item in all_sentences]

            for j in range(len(top3)):
                where.append(post_or_comment_or_reply(top3[j][0]))
                result.append([index, post, comment, score, all_tokens[top3[j][0]], top3[j][2],
                               post_or_comment_or_reply(top3[j][0])])

            # result.append([index, post, comment, score, all_tokens[top3[0][0]], top3[0][2], where[0]])
            # result.append([index, post, comment, score, all_tokens[top3[1][0]], top3[1][2], where[1]])
            # result.append([index, post, comment, score, all_tokens[top3[2][0]], top3[2][2], where[2]])

            return result, label, torch.argmax(pred).item()


def main(true_label, pred_label, target):
    post_df = pd.read_csv('../csv/dataset/liwc_post.csv', encoding='UTF-8')
    comment_df = pd.read_csv('../csv/dataset/liwc_comment.csv', encoding='UTF-8')
    reply_df = pd.read_csv('../csv/dataset/avg_satisfaction_raw_0-999.csv', encoding='ISO-8859-1')

    modes = [['seg', 'seg', 'snt']]

    # satisfaction score (y)
    satisfactions_float = list(reply_df['satisfy_composite'])
    satisfactions = []

    for s in satisfactions_float:
        if s < 3.5:
            satisfactions.append(0)
        elif s < 5:
            satisfactions.append(1)
        else:
            satisfactions.append(2)

    reply_contents = list(reply_df['replyContent'])
    post_contents = list(post_df['content'])
    comment_bodies = list(comment_df['content'])

    for mode in modes:
        post_sequences = get_sequences(post_contents, mode[0])
        comment_sequences = get_sequences(comment_bodies, mode[1])
        reply_sequences = get_sequences(reply_contents, mode[2])

        data = []
        max_post, max_comment, max_reply = 0, 0, 0
        i = 0
        for post, comment, reply, satisfaction, satisfaction_float in zip(post_sequences, comment_sequences,
                                                                          reply_sequences, satisfactions,
                                                                          satisfactions_float):
            if len(post) > max_post:
                max_post = len(post)
            if len(comment) > max_comment:
                max_comment = len(comment)
            if len(reply) > max_reply:
                max_reply = len(reply)

            data.append([i, post, comment, reply, satisfaction, satisfaction_float])
            i += 1

        max_count = max(max_post, max_comment, max_reply)
        columns = ['index', 'post_contents', 'comment_contents', 'reply_contents', 'label', 'score']
        df = pd.DataFrame(data, columns=columns)

        # data split (train & test sets)
        idx_train, idx_remain = train_test_split(df.index.values, test_size=0.20, random_state=42)
        idx_val, idx_test = train_test_split(idx_remain, test_size=0.50, random_state=42)

        train_df = df.iloc[idx_train]
        val_df = df.iloc[idx_val]
        test_df = df.iloc[idx_test]

        count_min_label = min(train_df['label'].value_counts())

        labels = [0, 1, 2]

        train_sample_df = pd.DataFrame([], columns=columns)

        for label in labels:
            tmp = train_df[train_df['label'] == label]
            tmp_sampled = tmp.sample(frac=1).iloc[:count_min_label]
            train_sample_df = pd.concat([train_sample_df, tmp_sampled])

        train_sample_df = train_sample_df.sample(frac=1)

    index_list = get_indexes(f'../csv/splitbert_classifier/post_comment/seg_seg/epoch_4_result.csv')
    device, pc_model, tokenizer = prepare_model(labels, max_count, max_post, max_comment,
                                                '../splitbert/model/seg_seg/attention', 5, False, 'diagonal', True)

    for i in index_list:
        splitbert_integrated_gradient(i, labels, pc_model, device, tokenizer, max_count, post_sequences[i],
                                      comment_sequences[i], post_contents[i], comment_bodies[i], satisfactions[i],
                                      satisfactions_float[i], true_label, pred_label, target, True)