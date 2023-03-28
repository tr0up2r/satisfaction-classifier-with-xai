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
from transformers import modeling_outputs
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import r2_score
import csv

import sys
sys.path.append('../custom_transformers')


def split_df(df):
    inputs_train, inputs_test, IS_labels_train, IS_labels_test, ES_labels_train, ES_labels_test = train_test_split(
        df.index.values,
        df.IS_label.values,
        df.ES_label.values,
        test_size=test_size,
        random_state=seed)

    df['data_type'] = ['not_set'] * df.shape[0]

    df.loc[inputs_train, 'data_type'] = 'train'
    df.loc[inputs_test, 'data_type'] = 'test'

    return df, inputs_train, inputs_test, IS_labels_train, IS_labels_test, ES_labels_train, ES_labels_test


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
    labels_train = torch.tensor(labels_train)

    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(labels_test)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train, labels_train)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test, labels_test)

    dataloader_train = DataLoader(dataset_train, sampler=SequentialSampler(dataset_train), batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size)

    return dataloader_train, dataloader_test


class BertForMultiValueRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.classifier1 = nn.Linear(config.hidden_size, config.hidden_size//2)
        self.classifier2 = nn.Linear(config.hidden_size//2, 1)

        # Initialize weights and apply final processing
        self.post_init()

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
        satisfaction_model_mode: Optional[bool] = None,
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
        output = self.classifier1(pooled_output)

        if satisfaction_model_mode:
            return output

        logits = self.classifier2(output)

        if not labels:
            return logits

        else:
            loss = None
            self.config.problem_type = "regression"
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze().to(torch.float32), labels.squeeze().to(torch.float32))

            return modeling_outputs.SequenceClassifierOutput(
                loss=loss,
                logits=logits
            )
        '''
        loss = None
        self.config.problem_type = "regression"
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze().to(torch.float32), labels.squeeze().to(torch.float32))

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
        '''


def train_model_and_save(dataloader_train, dataloader_val, target):
    model = BertForMultiValueRegression.from_pretrained('bert-base-uncased', num_labels=1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    epochs = 50
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    def evaluate(dataloader_val):
        model.eval()
        loss_val_total = 0

        v_predictions, v_true_vals = [], []

        for v_batch in dataloader_val:
            v_batch = tuple(b.to(device) for b in v_batch)

            v_inputs = {'input_ids': v_batch[0],
                        'attention_mask': v_batch[1],
                        'labels': v_batch[2],  # same in both batches.
                        'satisfaction_model_mode': False
                      }

            with torch.no_grad():
                v_outputs = model(**v_inputs)

            v_loss = v_outputs[0]
            v_logits = v_outputs[1]
            loss_val_total += v_loss.item()

            v_logits = v_logits.detach().cpu().numpy()
            v_label_ids = v_inputs['labels'].cpu().numpy()
            v_predictions.append(v_logits)
            v_true_vals.append(v_label_ids)

        loss_val_avg = loss_val_total / len(dataloader_val)

        v_predictions = np.concatenate(v_predictions, axis=0)
        v_true_vals = np.concatenate(v_true_vals, axis=0)

        # print(pooled_outputs)

        return loss_val_avg, v_predictions, v_true_vals

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
            satisfaction_model_mode = False
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],  # same in both batches.
                      'satisfaction_model_mode': satisfaction_model_mode
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
        val_loss, predictions, true_vals = evaluate(dataloader_val)
        evaluation_result.append([val_loss, predictions, torch.tensor(true_vals)])
        print(f'Validation loss: {val_loss}')

        true_vals = evaluation_result[0][2].tolist()
        predict = evaluation_result[0][1].tolist()

        tqdm.write(f'R^2 score: {r2_score(true_vals, predict)}')

        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(
            f'../predicting-satisfaction-using-graphs/csv/IS_ES_regressor/batch_{batch_size}_lr_1e-5/model2/{target}/epoch_{epoch}_predicted_vals.csv')

        training_result.append([epoch, loss_train_avg, val_loss, r2_score(true_vals, predict)])
        torch.save(model.state_dict(), f'../predicting-satisfaction-using-graphs/csv/IS_ES_regressor/model2/{target}/epoch_{epoch}.model')

    fields = ['epoch', 'training_loss', 'validation_loss', 'r^2_score']

    with open(
            f'../predicting-satisfaction-using-graphs/csv/IS_ES_regressor/batch_{batch_size}_lr_1e-5/model2/{target}/training_result.csv',
            'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(training_result)

    true_df = pd.DataFrame(true_vals)
    true_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/{target}_true_vals.csv')


if __name__ == '__main__':
    data_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/seeker_satisfaction_1000_thread.csv',
                          encoding='ISO-8859-1')
    post_contents = list(data_df['post_content'])
    comment_bodies = list(data_df['comment_body'])
    post_IS = list(data_df['post_IS'])
    post_ES = list(data_df['post_ES'])
    comment_IS = list(data_df['comment_IS'])
    comment_ES = list(data_df['comment_ES'])

    post_data = []
    comment_data = []

    for content, IS, ES in zip(post_contents, post_IS, post_ES):
        if content != '[deleted]' and content != '[removed]':
            post_data.append([content, IS, ES])

    for body, IS, ES in zip(comment_bodies, comment_IS, comment_ES):
        if body != '[deleted]' and body != '[removed]':
            comment_data.append([body, IS, ES])

    columns = ['contents', 'IS_label', 'ES_label']

    post_df = pd.DataFrame(post_data, columns=columns)
    comment_df = pd.DataFrame(comment_data, columns=columns)

    dfs = [post_df, comment_df]

    test_size = 0.2
    seed = 42

    post_df, post_inputs_train, post_input_test, post_IS_labels_train, post_IS_labels_test, post_ES_labels_train, post_ES_labels_test = split_df(
        post_df)
    comment_df, comment_inputs_train, comment_input_test, comment_IS_labels_train, comment_IS_labels_test, comment_ES_labels_train, comment_ES_labels_test = split_df(
        comment_df)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    batch_size = 3

    post_IS_dataloader_train, post_IS_dataloader_test = generate_dataloader(post_df, post_IS_labels_train,
                                                                            post_IS_labels_test, batch_size)
    post_ES_dataloader_train, post_ES_dataloader_test = generate_dataloader(post_df, post_ES_labels_train,
                                                                            post_ES_labels_test, batch_size)
    comment_IS_dataloader_train, comment_IS_dataloader_test = generate_dataloader(comment_df, comment_IS_labels_train,
                                                                                  comment_IS_labels_test, batch_size)
    comment_ES_dataloader_train, comment_ES_dataloader_test = generate_dataloader(comment_df, comment_ES_labels_train,
                                                                                  comment_ES_labels_test, batch_size)

    train_list = [post_IS_dataloader_train, post_ES_dataloader_train, comment_IS_dataloader_train,
                  comment_ES_dataloader_train]
    test_list = [post_IS_dataloader_test, post_ES_dataloader_test, comment_IS_dataloader_test,
                 comment_ES_dataloader_test]

    targets = ['post_IS', 'post_ES', 'comment_IS', 'comment_ES']
    for train_dl, test_dl, target in zip(train_list, test_list, targets):
        print(target)
        train_model_and_save(train_dl, train_dl, target)