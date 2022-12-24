import pandas as pd
import numpy as np
import IS_ES_separate_regressor
import math
import gc

import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertPreTrainedModel
from transformers import BertModel
from transformers import BertForSequenceClassification
from typing import Optional, Tuple, Union
from torch.nn import MSELoss
from transformers import modeling_outputs
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import r2_score
import csv


post_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/liwc_post.csv', encoding='UTF-8')
comment_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/liwc_comment.csv', encoding='UTF-8')

# texts (x)
post_contents = list(post_df['content'])
comment_bodies = list(comment_df['content'])

# LIWC features (x)
features = []
features.append(list(post_df['auxverb']))
features.append(list(post_df['negate']))
features.append(list(post_df['verb']))
features.append(list(post_df['negemo']))
features.append(list(post_df['anger']))
features.append(list(post_df['percept']))
features.append(list(post_df['focuspresent']))
features.append(list(post_df['relativ']))
features.append(list(post_df['death']))
features.append(list(post_df['swear']))
features.append(list(post_df['OtherP']))

features.append(list(comment_df['WC']))
features.append(list(comment_df['Clout']))
features.append(list(comment_df['Tone']))
features.append(list(comment_df['i']))
features.append(list(comment_df['you']))
features.append(list(comment_df['posemo']))
features.append(list(comment_df['social']))
features.append(list(comment_df['QMark']))

# satisfaction score (y)
satisfactions = list(post_df['satisfaction'])

post_data = []
comment_data = []
feature_data = []

i = 0
for content, body, satisfaction in zip(post_contents, comment_bodies, satisfactions):
    post_data.append([content, satisfaction, i])
    comment_data.append([body, satisfaction, i])
    i += 1

for i in range(len(satisfactions)):
    f = []
    for f_list in features:
        f.append(f_list[i])
    feature_data.append(f)

columns = ['contents', 'label', 'i']
post_df = pd.DataFrame(post_data, columns=columns)
comment_df = pd.DataFrame(comment_data, columns=columns)

dfs = [post_df, comment_df]

test_size = 0.2
seed = 42


def split_df(df):
    inputs_train, inputs_test, labels_train, labels_test, index_train, index_test = train_test_split(
        df.index.values,
        df.label.values,
        df.i.values,
        test_size=test_size,
        random_state=seed)

    df['data_type'] = ['not_set'] * df.shape[0]

    df.loc[inputs_train, 'data_type'] = 'train'
    df.loc[inputs_test, 'data_type'] = 'test'

    return df, inputs_train, inputs_test, labels_train, labels_test, index_train, index_test


post_df, post_inputs_train, post_inputs_test, post_labels_train, post_labels_test, post_index_train, post_index_test = split_df(post_df)
comment_df, comment_inputs_train, comment_inputs_test, comment_labels_train, comment_labels_test, comment_index_train, comment_index_test = split_df(comment_df)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def generate_dataloader(df, labels_train, labels_test, index_train, index_test, batch_size):
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
    index_train = torch.tensor(index_train, dtype=torch.float32)

    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(labels_test, dtype=torch.float32)
    index_test = torch.tensor(index_test, dtype=torch.float32)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train, index_train)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test, index_test)

    dataloader_train = DataLoader(dataset_train, sampler=SequentialSampler(dataset_train), batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test


batch_size = 8

post_dataloader_train, post_dataloader_test = generate_dataloader(post_df, post_labels_train, post_labels_test,
                                                                  post_index_train, post_index_test, batch_size)
comment_dataloader_train, comment_dataloader_test = generate_dataloader(comment_df, comment_labels_train,
                                                                        comment_labels_test,
                                                                        comment_index_train, comment_index_test,
                                                                        batch_size)


class ConcatBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert_post = BertModel(config)
        self.bert_comment = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.classifier2 = nn.Linear(config.hidden_size, 100)
        self.classifier3 = nn.Linear(100+len(feature_data[0]), 1)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

        # Initialize weights and apply final processing
        # self.init()

    def forward(
        self,
        post_input_ids: Optional[torch.Tensor] = None,
        post_attention_mask: Optional[torch.Tensor] = None,
        comment_attention_mask: Optional[torch.Tensor] = None,
        comment_input_ids: Optional[torch.Tensor] = None,
        comment_vote: Optional[torch.Tensor] = None,
        index: Optional[torch.Tensor] = None,
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

        print(index)
        post_outputs = self.bert_post(
            post_input_ids,
            attention_mask=post_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        post_pooled_output = post_outputs[1]

        comment_outputs = self.bert_comment(
            comment_input_ids,
            attention_mask=comment_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        comment_pooled_output = comment_outputs[1]

        concat_result = self.classifier(torch.cat((post_pooled_output, comment_pooled_output), dim=1))  # should be b * (768*2) -> b * 768
        concat_result = self.dropout(concat_result)
        concat_result = self.relu(concat_result)

        concat_result = self.classifier2(concat_result)  # b * (768) -> b * 100
        concat_result = self.dropout(concat_result)
        concat_result = self.relu(concat_result)

        # add feature to tensor (concat result)
        new_concat_result = []
        for i in range(len(concat_result)):
            concat_list = concat_result[i].tolist()
            concat_list.extend(feature_data[int(index[i].item())])
            new_concat_result.append(torch.Tensor(concat_list))

        new_concat_result = torch.stack(new_concat_result, dim=0).to('cuda')

        output = self.classifier3(new_concat_result)
        print(output)

        loss = None
        self.config.problem_type = "regression"
        loss_fct = MSELoss()
        loss = loss_fct(output.squeeze(), labels.squeeze())

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=output
        )


model = ConcatBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)


# print(model.bert.config.hidden_size)
# print(model)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-8)

epochs = 200

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(post_dataloader_train) * epochs)

# loss_function = nn.MSELoss()


def evaluate(post_dataloader_val, comment_dataloader_val):
    model.eval()
    loss_val_total = 0

    predictions, true_vals = [], []

    with torch.no_grad():
        for p_batch, c_batch in zip(post_dataloader_val, comment_dataloader_val):
            torch.cuda.empty_cache()
            gc.collect()
            p_batch = tuple(b.to(device) for b in p_batch)
            c_batch = tuple(b.to(device) for b in c_batch)

            inputs = {'post_input_ids': p_batch[0],
                      'post_attention_mask': p_batch[1],
                      'comment_attention_mask': c_batch[1],
                      'comment_input_ids': c_batch[0],
                      'labels': p_batch[2],  # same in both batches.
                      'index': p_batch[3]
                      }

            outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(post_dataloader_val)

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

    for p_batch, c_batch in zip(post_dataloader_train, comment_dataloader_train):
        model.zero_grad()
        p_batch = tuple(b.to(device) for b in p_batch)
        c_batch = tuple(b.to(device) for b in c_batch)

        inputs = {'post_input_ids': p_batch[0],
                  'post_attention_mask': p_batch[1],
                  'index': p_batch[2],
                  'comment_attention_mask': c_batch[1],
                  'comment_input_ids': c_batch[0],
                  'labels': p_batch[2],  # same in both batches.
                  'index': p_batch[3],  # same in both batches.
                  }

        torch.cuda.empty_cache()
        gc.collect()
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
    loss_train_avg = loss_train_total / len(post_dataloader_train)
    print(f'Training loss: {loss_train_avg}')
    val_loss, predictions, true_vals = evaluate(post_dataloader_test, comment_dataloader_test)
    evaluation_result.append([val_loss, predictions, torch.tensor(true_vals)])
    print(f'Validation loss: {val_loss}')

    true_vals = evaluation_result[0][2].tolist()
    predict = sum(evaluation_result[0][1].tolist(), [])
    tqdm.write(f'R^2 score: {r2_score(true_vals, predict)}')

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/liwc_regressor/batch_{batch_size}_lr_1e-4/epoch_{epoch}_predicted_vals.csv')

    training_result.append([epoch, loss_train_avg, val_loss, r2_score(true_vals, predict)])


fields = ['epoch', 'training_loss', 'validation_loss', 'r^2_score']

with open(f'../predicting-satisfaction-using-graphs/csv/liwc_regressor/batch_{batch_size}_lr_1e-4/training_result.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(training_result)

true_df = pd.DataFrame(true_vals)
true_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/liwc_regressor/batch_{batch_size}_lr_1e-4/true_vals.csv')