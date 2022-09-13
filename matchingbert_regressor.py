from spacy.lang.en import English
import pandas as pd
import numpy as np
import math

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


nlp = English()
nlp.add_pipe("sentencizer")
doc = nlp("This is a sentence. Is this a sentence? This is another sentence.")
print(len(list(doc.sents)))

result_df = pd.read_csv('seeker_satisfaction_1000_thread.csv')
# print(result_df)
post_ids = list(result_df['post_id'])
post_contents = list(result_df['post_content'])
post_titles = list(result_df['post_title'])

post_ids_contents = []
post_ids_titles = []

for i in range(len(post_ids)):
    if post_contents[i] != '[removed]' and post_contents[i] != '[deleted]':
        post_ids_contents.append([post_ids[i], post_contents[i]])
    if post_titles[i] != '[deleted by user]':
        post_ids_titles.append([post_ids[i], post_titles[i]])

comment_ids = list(result_df['comment_id'])
comment_bodies = list(result_df['comment_body'])

parent_ids_comment_ids_bodies = []

for i in range(len(comment_ids)):
    if comment_bodies[i] != '[removed]' and comment_bodies[i] != '[deleted]':
        parent_ids_comment_ids_bodies.append([post_ids[i], comment_ids[i], comment_bodies[i]])

post_contents_len_list = []
comment_bodies_len_list = []

for p_pair, c_pair in zip(post_ids_contents, parent_ids_comment_ids_bodies):
    post_contents_len_list.append(len(list(nlp(p_pair[1]).sents)))
    comment_bodies_len_list.append(len(list(nlp(c_pair[2]).sents)))

print(post_ids_contents[0][1])

print(post_contents_len_list)
print(comment_bodies_len_list)


data_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/seeker_satisfaction_1000_thread.csv', encoding='ISO-8859-1')
# post_titles = list(data_df['post_title'])
post_contents = list(data_df['post_content'])
comment_bodies = list(data_df['comment_body'])
satisfactions = list(data_df['satisfaction'])

post_data = []
comment_data = []
k = 3

for content, body, satisfaction in zip(post_contents, comment_bodies, satisfactions):
    count_label = 0
    if content != '[deleted]' and content != '[removed]' and body != '[deleted]' and body != '[removed]':
        p_sentence_count = len(list(nlp(content).sents))
        p_sentences = list(nlp(content).sents)
        for i in range(p_sentence_count):
            p_sentences[i] = str(p_sentences[i])

        c_sentence_count = len(list(nlp(body).sents))
        c_sentences = list(nlp(body).sents)
        for i in range(c_sentence_count):
            c_sentences[i] = str(c_sentences[i])

        if p_sentence_count >= 3:
            content = []
            for i in range(k):
                if i == k-1:
                    content.append(' '.join(p_sentences[round(i*(p_sentence_count / k)):]))
                else:
                    content.append(' '.join(p_sentences[round(i*(p_sentence_count / k)):round((i+1)*(p_sentence_count / k))]))
            for i in range(len(content)):
                post_data.append([content[i], satisfaction, i+1, len(list(nlp(content[i]).sents))])
        else:
            post_data.append([content, satisfaction, 0, len(p_sentences)])

        if c_sentence_count >= 3:
            body = []
            for i in range(k):
                if i == k - 1:
                    body.append(' '.join(c_sentences[round(i*(c_sentence_count / k)):]))
                else:
                    body.append(' '.join(c_sentences[round(i*(c_sentence_count / k)):round((i+1)*(c_sentence_count / k))]))
            for i in range(len(body)):
                comment_data.append([body[i], satisfaction, i+1, len(body[i])])
        else:
            comment_data.append([body, satisfaction, 0, len(c_sentences)])


def make_df(data_list, index):
    new_list = []
    for data in data_list:
        if data[2] == index:
            new_list.append(data)

    return pd.DataFrame(new_list, columns=['contents', 'label', 'group', 'sentence_count'])

p_dfs = []
c_dfs = []
for i in range(4):
    p_dfs.append(make_df(post_data, i))
    c_dfs.append(make_df(comment_data, i))


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

    return [df, labels_train, labels_test]


for_dl_p = []
for_dl_c = []

for p_df, c_df in zip(p_dfs, c_dfs):
    for_dl_p.append(split_df(p_df))
    for_dl_c.append(split_df(c_df))

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
pc_loader_pairs_train = []
pc_loader_pairs_test = []

for p, c in zip(for_dl_p, for_dl_c):
    print(len(p))
    post_dataloader_train, post_dataloader_test = generate_dataloader(p[0], p[1], p[2], batch_size)
    comment_dataloader_train, comment_dataloader_test = generate_dataloader(c[0], c[1], c[2], batch_size)
    pc_loader_pairs_train.append([post_dataloader_train, comment_dataloader_train])
    pc_loader_pairs_test.append([post_dataloader_test, comment_dataloader_test])

print('ok?')


class MatchingBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size*3, 1)
        self.classifier2 = nn.Linear(2, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        post_input_ids1: Optional[torch.Tensor] = None,
        post_input_ids2: Optional[torch.Tensor] = None,
        post_input_ids3: Optional[torch.Tensor] = None,
        post_attention_mask1: Optional[torch.Tensor] = None,
        post_attention_mask2: Optional[torch.Tensor] = None,
        post_attention_mask3: Optional[torch.Tensor] = None,
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

        output_list = []
        ids_list = [post_input_ids1, post_input_ids2, post_input_ids3]
        attention_list = [post_attention_mask1, post_attention_mask2, post_attention_mask3]

        for id, attention in zip(ids_list, attention_list):
            post_outputs = self.bert(
                id,
                attention_mask=attention,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output_list.append(post_outputs[1])

        logits = self.classifier(torch.cat((output_list[0], output_list[1], output_list[2]), dim=1))  # should be b * (768*2) -> b * 1

        loss = None
        self.config.problem_type = "regression"
        loss_fct = MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


model = MatchingBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

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
                                            num_training_steps=len(post_dataloader_train) * epochs)

# loss_function = nn.MSELoss()

for p_batch1, p_batch2, p_batch3 in zip(pc_loader_pairs_train[1][0], pc_loader_pairs_train[2][0], pc_loader_pairs_train[3][0]):
    p_batch1 = tuple(b.to(device) for b in p_batch1)
    p_batch2 = tuple(b.to(device) for b in p_batch2)
    p_batch3 = tuple(b.to(device) for b in p_batch3)

    inputs = {'post_input_ids1': p_batch1[0],
              'post_input_ids2': p_batch2[0],
              'post_input_ids3': p_batch3[0],
              'post_attention_mask1': p_batch1[1],
              'post_attention_mask2': p_batch2[1],
              'post_attention_mask3': p_batch3[1],
              'labels': p_batch1[2]  # same in both batches.
              }

    outputs = model(**inputs)
    loss = outputs[0]
    print(outputs[1])

    '''
    print(f'i : {i}, loss : {loss}')
    loss_train_total += loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    i += 1
    
def evaluate(post_dataloader_val, comment_dataloader_val):
    model.eval()
    loss_val_total = 0

    predictions, true_vals = [], []

    for p_batch, c_batch in zip(post_dataloader_val, comment_dataloader_val):
        p_batch = tuple(b.to(device) for b in p_batch)
        c_batch = tuple(b.to(device) for b in c_batch)

        inputs = {'post_input_ids': p_batch[0],
                  'comment_input_ids': c_batch[0],
                  'post_attention_mask': p_batch[1],
                  'comment_attention_mask': c_batch[1],
                  'labels': p_batch[2]  # same in both batches.
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
                  'comment_input_ids': c_batch[0],
                  'post_attention_mask': p_batch[1],
                  'comment_attention_mask': c_batch[1],
                  'labels': p_batch[2]  # same in both batches.
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
    loss_train_avg = loss_train_total / len(post_dataloader_train)
    print(f'Training loss: {loss_train_avg}')
    val_loss, predictions, true_vals = evaluate(post_dataloader_test, comment_dataloader_test)
    evaluation_result.append([val_loss, predictions, torch.tensor(true_vals)])
    print(f'Validation loss: {val_loss}')

    true_vals = evaluation_result[0][2].tolist()
    predict = sum(evaluation_result[0][1].tolist(), [])
    tqdm.write(f'R^2 score: {r2_score(true_vals, predict)}')

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/concatbert/batch_{batch_size}_lr_1e-5/epoch_{epoch}_predicted_vals.csv')

    training_result.append([epoch, loss_train_avg, val_loss, r2_score(true_vals, predict)])


fields = ['epoch', 'training_loss', 'validation_loss', 'r^2_score']

with open(f'../predicting-satisfaction-using-graphs/csv/concatbert/batch_{batch_size}_lr_1e-5/training_result.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(training_result)

true_df = pd.DataFrame(true_vals)
true_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/true_vals.csv')
'''