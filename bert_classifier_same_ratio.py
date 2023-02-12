import pandas as pd
import numpy as np
import random
import csv

import torch
#from tqdm.notebook import tqdm
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score


post_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/liwc_post.csv', encoding='UTF-8')
comment_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/liwc_comment.csv', encoding='UTF-8')

# texts (x)
post_contents = list(post_df['content'])
comment_bodies = list(comment_df['content'])

# satisfaction score (y)
satisfactions_float = list(post_df['satisfaction'])
satisfactions = []

for s in satisfactions_float:
    if s < 3.5:
        satisfactions.append(0)
    elif s < 5:
        satisfactions.append(1)
    else:
        satisfactions.append(2)

data = []

for content, body, satisfaction in zip(post_contents, comment_bodies, satisfactions):
    data.append([content + '[SEP]' + body, satisfaction])

columns = ['contents', 'label']
df = pd.DataFrame(data, columns=columns)

# data split (train & test sets)
idx_train, idx_remain = train_test_split(df.index.values, test_size=0.20, random_state=42)
idx_val, idx_test = train_test_split(idx_remain, test_size=0.50, random_state=42)

print(idx_train.shape, idx_val.shape, idx_test.shape)

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


# BERT
# uncased - upper case 문자들을 lower case로 바꾸고 난 후 tokenize한 모델.
# do_lower_case - upper case를 lower case로 변환할 것인지 여부를 정하는 param.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)

encoded_data_train = tokenizer.batch_encode_plus(
    train_sample_df.contents.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    val_df.contents.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(train_sample_df.label.values.astype(int))

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(val_df.label.values.astype(int))

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# Bert 모델로는 BertForSequenceClassification 사용.
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=len(labels),
                                                      problem_type='multi_label_classification',
                                                      output_attentions=False,
                                                      output_hidden_states=False)


# Data Loaders
batch_size = 4

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val,
                                   sampler=SequentialSampler(dataset_val),
                                   batch_size=batch_size)

# Optimizer & Scheduler
# 사용할 optimizer : AdamW
optimizer = AdamW(model.parameters(),
                  lr=1e-5,  # learning rate.
                  eps=1e-8)  # learning rate가 0으로 나눠지는 것을 방지하기 위한 epsilon 값.

# epochs 5로 했더니 overfitting 되는 듯.
# 3 정도가 적당?
epochs = 30

# learning rate decay를 위한 scheduler. (linear 이용)
# lr이 0부터 optimizer에서 설정한 lr까지 linear하게 warmup 됐다가 다시 0으로 linear 하게 감소.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train) * epochs)


# Performance Metrics


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


# Training Loop
device = torch.device('cuda')
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0

    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        one_hot_labels = torch.nn.functional.one_hot(batch[2], num_classes=len(labels))

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': one_hot_labels.type(torch.float)
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = batch[2].cpu().numpy()

        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    accuracy_per_class(predictions, true_vals)

    return loss_val_avg, predictions, true_vals


model.to(device)

training_result = []

for epoch in tqdm(range(1, epochs + 1)):
    evaluation_result = []
    model.train()
    loss_train_total = 0
    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    i = 0
    for batch in progress_bar:
        model.zero_grad()
        # batch를 device(cpu)에 넣음.
        batch = tuple(b.to(device) for b in batch)
        one_hot_labels = torch.nn.functional.one_hot(batch[2], num_classes=len(labels))

        # batch에서 data 추출.
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': one_hot_labels.type(torch.float),
                  }
        # Forward 수행.
        outputs = model(**inputs)
        # loss 구함.
        loss = outputs[0]
        #print(f'i : {i}, loss : {loss}')
        # 총 loss 계산.
        loss_train_total += loss.item()
        loss.backward()
        # gradient clipping을 진행.
        # gradient exploding을 방지하기 위함으로,
        # gradient가 일정 threshold를 넘어가면 clipping을 해준다.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # gradient를 이용해 weight update.
        optimizer.step()
        # scheduler를 이용해 learning rate 조절.
        scheduler.step()
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
        i += 1
    tqdm.write(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1_macro, val_f1_micro = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Macro, Micro): {val_f1_macro}, {val_f1_micro}')
    training_result.append([epoch, loss_train_avg, val_loss, val_f1_macro, val_f1_micro])

fields = ['epoch', 'training_loss', 'validation_loss', 'f1_score_macro', 'f1_score_micro']

with open(
        f'../predicting-satisfaction-using-graphs/csv/bert_classifier/training_result.csv',
        'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(training_result)
