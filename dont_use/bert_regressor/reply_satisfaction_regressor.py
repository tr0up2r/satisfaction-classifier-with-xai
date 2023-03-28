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
from sklearn.metrics import r2_score
import csv

import sys
sys.path.append('../custom_transformers')


data_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/seeker_satisfaction_1000_thread.csv', encoding='ISO-8859-1')
reply_contents = list(data_df['reply_content'])
post_contents = list(data_df['post_content'])
comment_bodies = list(data_df['comment_body'])
satisfactions = list(data_df['satisfaction'])

reply_data = []

for reply, content, body, satisfaction in zip(reply_contents, post_contents, comment_bodies, satisfactions):
    if content != '[deleted]' and content != '[removed]' and body != '[deleted]' and body != '[removed]':
        reply_data.append([reply, satisfaction])

df = pd.DataFrame(reply_data, columns=['contents', 'label'])

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


df, inputs_train, input_test, labels_train, labels_test = split_df(df)

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

    dataloader_train = DataLoader(dataset_train, sampler=SequentialSampler(dataset_train), batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size)

    return dataloader_train, dataloader_test


batch_size = 3

dataloader_train, dataloader_test = generate_dataloader(df, labels_train, labels_test, batch_size)


class CustomBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.classifier1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.classifier2 = nn.Linear(config.hidden_size // 2, 1)
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
        self.config.problem_type = "regression"
        loss_fct = MSELoss()
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

epochs = 200
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train) * epochs)

# loss_function = nn.MSELoss()


def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0

    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]
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

    return loss_val_avg, predictions, true_vals


training_result = []
val_values = []

for epoch in tqdm(range(1, epochs + 1)):
    evaluation_result = []
    model.train()
    loss_train_total = 0
    # print(dataloader_train)

    i = 0

    for batch in dataloader_train:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]  # same in both batches.
                  }

        outputs = model(**inputs)
        loss = outputs[0]
        print(f'i : {i}, loss : {loss}')
        loss_train_total += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        i += 1

    # torch.save(model.state_dict(), f'data_volume/inf_macro_finetuned_BERT_epoch_{epoch}.model')
    print(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total / len(dataloader_train)
    print(f'Training loss: {loss_train_avg}')
    val_loss, predictions, true_vals = evaluate(dataloader_test)
    evaluation_result.append([val_loss, predictions, torch.tensor(true_vals)])
    print(f'Validation loss: {val_loss}')

    true_vals = evaluation_result[0][2].tolist()
    predict = sum(evaluation_result[0][1].tolist(), [])
    tqdm.write(f'R^2 score: {r2_score(true_vals, predict)}')

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/reply_regressor/batch_{batch_size}_lr_1e-5/epoch_{epoch}_predicted_vals.csv')

    training_result.append([epoch, loss_train_avg, val_loss, r2_score(true_vals, predict)])


fields = ['epoch', 'training_loss', 'validation_loss', 'r^2_score']

with open(f'../predicting-satisfaction-using-graphs/csv/reply_regressor/batch_{batch_size}_lr_1e-5/training_result.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(training_result)

true_df = pd.DataFrame(true_vals)
true_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/true_vals.csv')