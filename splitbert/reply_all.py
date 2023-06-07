import numpy as np
import pandas as pd
import random
import csv

import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score


def preparing_data(df, label_dict):
    # data split (train & test sets)
    X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                      df.label.values,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=df.label.values)

    df['data_type'] = ['not_set'] * df.shape[0]

    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'

    # print(df.groupby(['inf', 'label', 'data_type']).count())

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == 'train'].comment.values,  # df[df.data_type == 'train'].{your_column_name}.values
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == 'val'].comment.values,  # df[df.data_type == 'val'].{your_column_name}.values
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type == 'train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type == 'val'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    return dataset_train, dataset_val


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels):

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class {label} : ', end='')
        print(f'{len(y_preds[y_preds == label])}/{len(y_true)}')


def evaluate(model, device, dataloader_val):
    model.eval()
    loss_val_total = 0

    predictions, true_vals = [], []

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

    return loss_val_avg, predictions, true_vals


def train(dataset_train, dataset_val, path):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=3,
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    for param in model.bert.parameters():
        param.requires_grad = False

    # adjust hyper parameters
    batch_size = 4
    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)
    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)
    optimizer = AdamW(model.parameters(),
                      lr=2e-4,
                      eps=1e-8)
    epochs = 10
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    # Training Loop
    device = torch.device('cuda')
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model.to(device)
    training_result = []

    for epoch in tqdm(range(1, epochs+1)):
        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        i = 0
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            outputs = model(**inputs)
            loss = outputs[0]
            # print(f'i : {i}, loss : {loss}')
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # print(progress_bar)
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
            i += 1

        torch.save(model.state_dict(), path + f'{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(model, device, dataloader_validation)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
        training_result.append([epoch, loss_train_avg, val_loss, val_f1])

        accuracy_per_class(predictions, true_vals)

    fields = ['epoch', 'training_loss', 'validation_loss', 'f1_score_macro', 'f1_score_micro']

    with open(path + f'/csv/splitbert_classifier/reply_all/training_result.csv', 'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(training_result)


if __name__ == "__main__":
    path = '/data1/mykim/predicting-satisfaction-using-graphs'
    reply_df = pd.read_csv(path + '/csv/dataset/avg_satisfaction_raw_0-999.csv', encoding='ISO-8859-1')
    reply_contents = list(reply_df['replyContent'])

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

    data = []

    i = 0
    for content, satisfaction in zip(reply_contents, satisfactions):
        data.append([i, content, satisfaction])
        i += 1

    columns = ['index', 'reply', 'label']
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    print(train_sample_df.label.values)

    encoded_data_train = tokenizer.batch_encode_plus(
        train_sample_df.reply.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        val_df.reply.values,  # df[df.data_type == 'val'].{your_column_name}.values
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    print(train_sample_df.label.astype(int))
    labels_train = torch.tensor(list(train_sample_df.label.astype(int)))

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(list(val_df.label.astype(int)))

    print(labels_val)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    train(dataset_train, dataset_val, path)
