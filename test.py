import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from transformers import BertConfig

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import r2_score
import csv

import sys
sys.path.append('../custom_transformers')

# from custom_transformers.src.transformers.models.bert import BertTokenizer
# from custom_transformers.src.transformers.models.bert import BertForSequenceClassification
# from custom_transformers.src.transformers.models.bert import BertConfig




data_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/seeker_satisfaction_1000_thread.csv', encoding='ISO-8859-1')
# post_titles = list(data_df['post_title'])
post_contents = list(data_df['post_content'])
comment_bodies = list(data_df['comment_body'])
satisfactions = list(data_df['satisfaction'])

post_data = []
comment_data = []

for content, body, satisfaction in zip(post_contents, comment_bodies, satisfactions):
    if content != '[deleted]' and content != '[removed]' and body != '[deleted]' and body != '[removed]':
        # data.append([content + ' ' + body, satisfaction])
        post_data.append([content, satisfaction])
        comment_data.append([body, satisfaction])

post_df = pd.DataFrame(post_data, columns=['contents', 'label'])
comment_df = pd.DataFrame(comment_data, columns=['contents', 'label'])

dfs = [post_df, comment_df]

test_size = 0.2
seed = 42


def split_df(df):
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(df.index.values,
                                                                            df.label.values,
                                                                            test_size=test_size,
                                                                            random_state=seed)

    df['data_type'] = ['not_set'] * df.shape[0]

    df.loc[inputs_train, 'data_type'] = 'train'
    df.loc[inputs_test, 'data_type'] = 'test'

    return df, inputs_train, inputs_test, labels_train, labels_test


post_df, post_inputs_train, post_input_test, post_labels_train, post_labels_test = split_df(post_df)
comment_df, comment_inputs_train, comment_input_test, comment_labels_train, comment_labels_test = split_df(comment_df)


print(post_df.loc[0])
print(comment_df.loc[0])


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def generate_dataloader(df, labels_train, labels_test, batch_size):
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == 'train'].contents.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=512,
        return_tensors='pt'
    )

    encoded_data_test = tokenizer.batch_encode_plus(
        df[df.data_type == 'test'].contents.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=512,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(labels_train, dtype=torch.float32)

    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(labels_test, dtype=torch.float32)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size)

    return dataloader_train, dataloader_test


post_dataloader_train, post_dataloader_test = generate_dataloader(post_df, post_labels_train, post_labels_test, 3)
comment_dataloader_train, comment_dataloader_test = generate_dataloader(comment_df, comment_labels_train, comment_labels_test, 3)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

for batch in post_dataloader_test:
    batch = tuple(b.to(device) for b in batch)

    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'labels': batch[2],
              }

    print(inputs['labels'])


for batch in comment_dataloader_test:
    batch = tuple(b.to(device) for b in batch)

    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'labels': batch[2],
              }

    print(inputs['labels'])