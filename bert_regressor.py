import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import r2_score
import csv


data_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/seeker_satisfaction_1000_thread.csv', encoding='ISO-8859-1')
# post_titles = list(data_df['post_title'])
post_contents = list(data_df['post_content'])
comment_bodies = list(data_df['comment_body'])
satisfactions = list(data_df['satisfaction'])

data = []

for content, body, satisfaction in zip(post_contents, comment_bodies, satisfactions):
    if content != '[deleted]' and content != '[removed]' and body != '[deleted]' and body != '[removed]':
        data.append([content + ' ' + body, satisfaction])
        # data.append([content + '[SEP]' + body, satisfaction])

df = pd.DataFrame(data, columns=['contents', 'label'])

test_size = 0.2
seed = 42
inputs_train, inputs_test, labels_train, labels_test = train_test_split(df.index.values,
                                                                        df.label.values,
                                                                        test_size=test_size,
                                                                        random_state=seed)

df['data_type'] = ['not_set'] * df.shape[0]

df.loc[inputs_train, 'data_type'] = 'train'
df.loc[inputs_test, 'data_type'] = 'test'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)

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

# BertTokenizer.build_inputs_with_special_tokens()

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(labels_train, dtype=torch.float32)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(labels_test, dtype=torch.float32)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

batch_size = 3
# batch_size = 32

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

dataloader_test = DataLoader(dataset_test,
                                   sampler=SequentialSampler(dataset_test),
                                   batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=1,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

model.to(device)

optimizer = AdamW(model.parameters(),
                  lr=1e-5,  # learning rate.
                  eps=1e-8)  # learning rate가 0으로 나눠지는 것을 방지하기 위한 epsilon 값.

epochs = 2

# learning rate decay를 위한 scheduler. (linear 이용)
# lr이 0부터 optimizer에서 설정한 lr까지 linear하게 warmup 됐다가 다시 0으로 linear 하게 감소.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train) * epochs)

loss_function = nn.MSELoss()


def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0

    pooled_outputs, predictions, true_vals = [], [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1]
                  }

        with torch.no_grad():
            outputs = model.bert(**inputs)

        pooled_output = outputs[1]
        pooled_output = model.dropout(pooled_output)

        for output in pooled_output:
            pooled_outputs.append(output.detach().tolist())

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    # print(pooled_outputs)

    return loss_val_avg, predictions, true_vals, pooled_outputs


training_result = []
val_values = []

for epoch in tqdm(range(1, epochs + 1)):
    evaluation_result = []
    model.train()
    loss_train_total = 0
    # print(dataloader_train)
    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    i = 0

    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        # print(inputs['labels'])
        # batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        # print(batch_labels[0][0])
        outputs = model(**inputs)
        # print(outputs[1])
        loss = outputs[0]
        print(f'i : {i}, loss : {loss}')
        # r2 = my_r2_score(outputs[1], batch[2])
        # print(r2)
        # print('======================')
        loss_train_total += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
        i += 1

    # torch.save(model.state_dict(), f'data_volume/inf_macro_finetuned_BERT_epoch_{epoch}.model')
    tqdm.write(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')
    val_loss, predictions, true_vals, pooled_outputs = evaluate(dataloader_test)
    print(pooled_outputs)
    evaluation_result.append([val_loss, predictions, torch.tensor(true_vals)])
    tqdm.write(f'Validation loss: {val_loss}')

    true_vals = evaluation_result[0][2].tolist()
    # print(true_vals)
    predict = sum(evaluation_result[0][1].tolist(), [])
    # print(predict)
    tqdm.write(f'R^2 score: {r2_score(true_vals, predict)}')

    pred_df = pd.DataFrame(predictions)
    # pred_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/whitespace/batch_{batch_size}_lr_2e-5/epoch_{epoch}_predicted_vals.csv')

    training_result.append([epoch, loss_train_avg, val_loss, r2_score(true_vals, predict)])


fields = ['epoch', 'training_loss', 'validation_loss', 'r^2_score']

'''

with open(f'../predicting-satisfaction-using-graphs/csv/whitespace/batch_{batch_size}_lr_2e-5/training_result.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(training_result)
'''

true_df = pd.DataFrame(true_vals)
true_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/true_vals.csv')

pooled_df = pd.DataFrame(pooled_outputs)
pooled_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/pooled_outputs.csv')