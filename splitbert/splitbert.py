import pandas as pd
import numpy as np
import random
import csv
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import TensorDataset
from transformers import BertModel
from transformers import BertPreTrainedModel

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import modeling_outputs
from typing import Optional, Tuple, Union, Any
from sklearn.metrics import f1_score

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # print(sum_mask)
    return sum_embeddings / sum_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print(pe.shape)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # print(pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_batch_embeddings(model, fc_layer, embedding_size, input_ids, attention_mask, sentence_count,
                         max_sentences, batch_embeddings):
    for i in range(len(input_ids)):
        sbert_outputs = torch.empty(size=(max_sentences, 1, embedding_size)).to('cuda')

        model_output = model(input_ids[i][0:sentence_count[i]],
                                    attention_mask=attention_mask[i][0:sentence_count[i]])

        model_output = mean_pooling(model_output, attention_mask[i][0:sentence_count[i]])

        for j in range(sentence_count[i].item()):
            sbert_outputs[j] = model_output[j]

        for j in range(sentence_count[i].item(), max_sentences):
            sbert_outputs[j] = torch.zeros(1, embedding_size).to('cuda')

        # nn.Linear
        sbert_outputs_fc = fc_layer(sbert_outputs)
        sbert_outputs_fc = sbert_outputs_fc.swapaxes(0, 1)
        batch_embeddings[i] = sbert_outputs_fc

    return batch_embeddings


def make_masks(max_sentences, count):
    mask = (torch.triu(torch.ones(max_sentences, max_sentences)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda')

    zeros = torch.zeros(1, count.item()).to('cuda')
    ones = torch.ones(1, max_sentences - count.item()).to('cuda')
    key_padding_mask = torch.cat([zeros, ones], dim=1).type(torch.bool)

    return mask, key_padding_mask


class SplitBertTransformerModel(nn.Module):

    def __init__(self, num_labels, embedding_size, max_sentences, max_len1, max_len2):
        super().__init__()
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.max_sentences = max_sentences
        self.max_len1 = max_len1
        self.max_len2 = max_len2

        self.sbert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.fc_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding_size, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.pe = PositionalEncoding(self.embedding_size, max_len=self.max_sentences)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier1 = nn.Linear(self.embedding_size, self.embedding_size//2)
        self.classifier2 = nn.Linear(self.embedding_size//2, self.num_labels)
        self.classifier3 = nn.Linear(self.embedding_size, self.num_labels)
        # self.post_init()

    def forward(
            self,
            labels: Optional[torch.tensor] = None,
            input_ids1: Optional[torch.tensor] = None,
            input_ids2: Optional[torch.tensor] = None,
            attention_mask1: Optional[torch.tensor] = None,
            attention_mask2: Optional[torch.tensor] = None,
            sentence_count1: Optional[torch.tensor] = None,
            sentence_count2: Optional[torch.tensor] = None,
            indexes: Optional[torch.tensor] = None):

        current_batch_size = len(input_ids1)

        batch_encoder_embeddings = torch.empty(size=(current_batch_size, 1, self.max_len1, self.embedding_size),
                                               requires_grad=True).to('cuda')
        batch_decoder_embeddings = torch.empty(size=(current_batch_size, 1, self.max_len2, self.embedding_size),
                                               requires_grad=True).to('cuda')

        batch_encoder_embeddings = get_batch_embeddings(self.sbert, self.fc_layer, self.embedding_size, input_ids1,
                                                        attention_mask1, sentence_count1, self.max_len1,
                                                        batch_encoder_embeddings)
        batch_decoder_embeddings = get_batch_embeddings(self.sbert, self.fc_layer, self.embedding_size, input_ids2,
                                                        attention_mask2, sentence_count2, self.max_len2,
                                                        batch_decoder_embeddings)

        outputs = torch.empty(size=(current_batch_size, self.embedding_size), requires_grad=True).to('cuda')

        for encoder_embeddings, decoder_embeddings, count1, count2, i in zip(batch_encoder_embeddings,
                                                                             batch_decoder_embeddings,
                                                                             sentence_count1,
                                                                             sentence_count2,
                                                                             range(current_batch_size)):
            encoder_embeddings = encoder_embeddings.swapaxes(0, 1)
            decoder_embeddings = decoder_embeddings.swapaxes(0, 1)

            # add positional encoding
            encoder_embeddings = self.pe(encoder_embeddings)

            # make masks
            src_mask, src_key_padding_mask = make_masks(self.max_len1, count1)
            tgt_mask, tgt_key_padding_mask = make_masks(self.max_len2, count2)

            encoder_output = self.encoder(encoder_embeddings,
                                          mask=src_mask,
                                          src_key_padding_mask=src_key_padding_mask)

            decoder_output = self.decoder(tgt=decoder_embeddings, memory=encoder_output,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask)

            decoder_output = torch.mean(decoder_output[:count2], dim=0).squeeze(0)

            outputs[i] = decoder_output

        outputs = self.classifier1(outputs)
        logits = self.classifier2(outputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        return (loss, logits, outputs)


class SplitBertEncoderModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.embedding_size = config.embedding_size
        self.max_sentences = config.max_sentences

        self.sbert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.fc_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.pe = PositionalEncoding(self.embedding_size, max_len=self.max_sentences)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier1 = nn.Linear(self.embedding_size, self.embedding_size//2)
        self.classifier2 = nn.Linear(self.embedding_size//2, config.num_labels)
        self.classifier3 = nn.Linear(self.embedding_size, config.num_labels)
        self.post_init()

    def forward(
            self,
            labels: Optional[torch.tensor] = None,
            input_ids: Optional[torch.tensor] = None,
            attention_mask: Optional[torch.tensor] = None,
            sentence_count: Optional[torch.tensor] = None,
            indexes: Optional[torch.tensor] = None
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.SequenceClassifierOutput]:

        current_batch_size = len(input_ids)
        batch_embeddings = torch.empty(size=(current_batch_size, 1, self.max_sentences, self.embedding_size),
                                       requires_grad=True).to('cuda')

        batch_embeddings = get_batch_embeddings(self.sbert, self.embedding_size, input_ids, attention_mask,
                                                sentence_count, self.max_sentences, batch_embeddings)

        outputs = torch.empty(size=(current_batch_size, self.embedding_size), requires_grad=True).to('cuda')

        for embeddings, count, i in zip(batch_embeddings, sentence_count, range(current_batch_size)):
            embeddings = embeddings.swapaxes(0, 1)

            # add positional encoding
            embeddings = self.pe(embeddings)

            # make masks
            src_mask, src_key_padding_mask = make_masks(self.max_sentences, count)

            output = self.encoder(embeddings, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            print(output.shape)

            # decoder_output = torch.mean(decoder_output[:c_count], dim=0).squeeze(0)

            # outputs[i] = decoder_output

        outputs = self.classifier1(outputs)
        logits = self.classifier2(outputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs
        )


def conduct_input_ids_and_attention_masks(tokenizer, str_values, label_values, score_values, index_values,
                                          max_sentences_list):
    tensor_datasets = []
    pc_input_ids = []
    pc_attention_masks = []
    pc_sentence_count = []

    for value, max_sentences in zip(str_values, max_sentences_list):
        input_ids_list = []
        attention_masks_list = []
        sentence_count_list = []
        for contents in value:
            result = tokenizer(contents,
                            pad_to_max_length=True, truncation=True, max_length=128, return_tensors='pt')

            input_ids = result['input_ids']
            attention_masks = result['attention_mask']

            sentence_count_list.append(len(input_ids))

            # add zero pads to make all tensors' dimension (max_sentences, 128)
            pad = (0, 0, 0, max_sentences-len(input_ids))
            input_ids = nn.functional.pad(input_ids, pad, "constant", 0)  # effectively zero padding
            attention_masks = nn.functional.pad(attention_masks, pad, "constant", 0)

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_masks)

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_masks = torch.stack(attention_masks_list, dim=0)
        sentence_counts = torch.tensor(sentence_count_list)

        pc_input_ids.append(input_ids)
        pc_attention_masks.append(attention_masks)
        pc_sentence_count.append(sentence_counts)

        print(input_ids.shape, attention_masks.shape, sentence_counts.shape)

    labels = torch.tensor(label_values.astype(int))
    scores = torch.tensor(score_values.astype(float))
    indexes = torch.tensor(index_values.astype(int))

    # 0: posts / 1: comments
    return TensorDataset(labels, pc_input_ids[0], pc_input_ids[1],
                         pc_attention_masks[0], pc_attention_masks[1],
                         pc_sentence_count[0], pc_sentence_count[1], scores, indexes)


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


def evaluate(dataloader_val, model, device, target, labels):
    model.eval()
    loss_val_total = 0

    embeddings, predictions, true_vals, true_scores, indexes = [], [], [], [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        print(batch[0], len(labels))
        one_hot_labels = torch.nn.functional.one_hot(batch[0], num_classes=len(labels))

        if target == 'post_comment':
            inputs = {'labels': one_hot_labels.type(torch.float),
                      'input_ids1': batch[1],
                      'input_ids2': batch[2],
                      'attention_mask1': batch[3],
                      'attention_mask2': batch[4],
                      'sentence_count1': batch[5],
                      'sentence_count2': batch[6],
                      'indexes': batch[7]
                      }
        else:
            inputs = {'labels': one_hot_labels.type(torch.float),
                      'input_ids': batch[1],
                      'attention_mask': batch[2],
                      'sentence_count': batch[3],
                      'indexes': batch[4]
                      }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        hidden_states = outputs[2]
        loss_val_total += loss.item()

        hidden_states = hidden_states.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        label_ids = batch[0].cpu().numpy()
        score_ids = batch[7].cpu().numpy()
        index_ids = batch[8].cpu().numpy()

        embeddings.append(hidden_states)
        predictions.append(logits)
        true_vals.append(label_ids)
        true_scores.append(score_ids)
        indexes.append(index_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    embeddings = np.concatenate(embeddings, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    true_scores = np.concatenate(true_scores, axis=0)
    indexes = np.concatenate(indexes, axis=0)

    print(true_vals)

    accuracy_per_class(predictions, true_vals)

    return loss_val_avg, embeddings, predictions, true_vals, true_scores, indexes


def train(model, device, dataset_train, dataset_val, labels, target, path):
    optimizer = AdamW(model.parameters(),
                      lr=2e-4,
                      eps=1e-8)
    epochs = 10
    batch_size = 4

    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)
    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    model.to(device)

    training_result = []

    # Training Loop
    for epoch in tqdm(range(1, epochs + 1)):
        result_for_tsne = []
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        i = 0
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            one_hot_labels = torch.nn.functional.one_hot(batch[0], num_classes=len(labels))

            if target == 'post_comment':
                inputs = {'labels': one_hot_labels.type(torch.float),
                          'input_ids1': batch[1],
                          'input_ids2': batch[2],
                          'attention_mask1': batch[3],
                          'attention_mask2': batch[4],
                          'sentence_count1': batch[5],
                          'sentence_count2': batch[6],
                          'indexes': batch[7]
                          }
            else:
                inputs = {'labels': one_hot_labels.type(torch.float),
                          'input_ids': batch[1],
                          'attention_mask': batch[2],
                          'sentence_count': batch[3],
                          'indexes': batch[4]
                          }

            # check parameters are training
            '''
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(name, param.requires_grad)
            '''

            outputs = model(**inputs)
            loss = outputs[0]
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            predictions = np.concatenate([logits], axis=0)

            # 총 loss 계산.
            loss_train_total += loss.item()
            loss.backward()
            # print(loss.requires_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # gradient를 이용해 weight update.
            optimizer.step()
            # scheduler를 이용해 learning rate 조절.
            scheduler.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
            i += 1
        # torch.save(model.state_dict(),
        #            f'../predicting-satisfaction-using-graphs/model/splitbert_classifier/epoch_{epoch}.model')
        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        val_loss, embeddings, predictions, true_vals, true_scores, indexes = evaluate(dataloader_validation, model,
                                                                                      device, target, labels)

        embeddings = embeddings.tolist()
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.to_csv(path + f'/csv/splitbert_classifier/epoch_{epoch}_embeddings.csv')
        preds_flat = np.argmax(predictions, axis=1).flatten()
        labels_flat = true_vals.flatten()
        scores_flat = true_scores.flatten()
        indexes_flat = indexes.flatten()
        # print(type(embeddings))

        val_f1_macro, val_f1_micro = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Macro, Micro): {val_f1_macro}, {val_f1_micro}')
        training_result.append([epoch, loss_train_avg, val_loss, val_f1_macro, val_f1_micro])

        print(len(embeddings), len(preds_flat), len(labels_flat), len(scores_flat), len(indexes_flat))

        tsne_df = pd.DataFrame({'prediction': preds_flat, 'label': labels_flat,
                                'score': scores_flat, 'index': indexes_flat})
        tsne_df.to_csv(path + f'/csv/splitbert_classifier/epoch_{epoch}_result_for_tsne.csv')

    fields = ['epoch', 'training_loss', 'validation_loss', 'f1_score_macro', 'f1_score_micro']

    with open(
            path + '/csv/splitbert_classifier/training_result.csv',
            'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(training_result)
