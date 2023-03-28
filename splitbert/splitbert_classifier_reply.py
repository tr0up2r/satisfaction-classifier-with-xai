import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertPreTrainedModel
from transformers import BertModel
from typing import Optional, Tuple, Union
from torch.nn import MSELoss
from transformers import modeling_outputs
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

import csv

path = '/data1/mykim/predicting-satisfaction-using-graphs'
data_df = pd.read_csv(path + '/csv/dataset/avg_satisfaction_raw_0-999.csv', encoding='ISO-8859-1')
reply_contents = list(data_df['replyContent'])
satisfactions_float = list(data_df['satisfy_composite'])
satisfactions = []

for s in satisfactions_float:
    if s < 3.5:
        satisfactions.append(0)
    elif s < 5:
        satisfactions.append(1)
    else:
        satisfactions.append(2)

# 0: 222, 1: 489, 2: 289

reply_data = []

for reply, satisfaction in zip(reply_contents, satisfactions):
    reply_data.append([reply, satisfaction])

columns = ['contents', 'label']
df = pd.DataFrame(reply_data, columns=columns)

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

print(train_sample_df.shape)
print(val_df.shape)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def generate_dataloader(train_df, val_df, batch_size):
    encoded_data_train = tokenizer.batch_encode_plus(
        train_df.contents.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=512,
        return_tensors='pt'
    )

    encoded_data_test = tokenizer.batch_encode_plus(
        val_df.contents.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=512,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(train_df.label.values.astype(int))

    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(val_df.label.values.astype(int))

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    dataloader_train = DataLoader(dataset_train, sampler=SequentialSampler(dataset_train), batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size)

    return dataloader_train, dataloader_test


batch_size = 3
dataloader_train, dataloader_test = generate_dataloader(train_sample_df, val_df, batch_size)


class CustomBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.classifier1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.classifier2 = nn.Linear(config.hidden_size // 2, len(labels))
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

        # Initialize weights and apply final processing
        # self.init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        output = self.classifier1(pooled_output)  # should be b * 768 -> b * (768 // 2)
        output = self.dropout(output)
        output = self.relu(output)

        logits = self.classifier2(output)

        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# print(model.bert.config.hidden_size)
# print(model)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

'''
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
'''

epochs = 10
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train) * epochs)

# loss_function = nn.MSELoss()

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='macro'), f1_score(labels_flat, preds_flat, average='micro')


def accuracy_per_class(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    print(len(preds_flat))
    labels_flat = labels.flatten()
    print(len(labels_flat))

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class {label} : ', end='')
        print(f'{len(y_preds[y_preds == label])}/{len(y_true)}')


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

    print(predictions)
    print(true_vals)

    # print(pooled_outputs)
    accuracy_per_class(predictions, true_vals)

    return loss_val_avg, predictions, true_vals


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
        one_hot_labels = torch.nn.functional.one_hot(batch[2], num_classes=len(labels))

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': one_hot_labels.type(torch.float)
                  }

        outputs = model(**inputs)
        loss = outputs[0]
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
    val_loss, predictions, true_vals = evaluate(dataloader_test)

    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_vals.flatten()

    val_f1_macro, val_f1_micro = f1_score_func(predictions, true_vals)

    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Macro, Micro): {val_f1_macro}, {val_f1_micro}')

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(path + f'/csv/reply_classifier/epoch_{epoch}_predicted_vals.csv')

    training_result.append([epoch, loss_train_avg, val_loss, val_f1_macro, val_f1_micro])


fields = ['epoch', 'training_loss', 'validation_loss', 'f1_score_macro', 'f1_score_micro']

with open(path + '/csv/reply_classifier/training_result.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(training_result)

true_df = pd.DataFrame(true_vals)
true_df.to_csv(path + '/csv/true_vals.csv')
