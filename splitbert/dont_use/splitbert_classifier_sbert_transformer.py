import pandas as pd
import numpy as np
import random
import csv
import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertModel
from transformers import BertPreTrainedModel

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import modeling_outputs
from typing import Optional, Tuple, Union
from sklearn.metrics import f1_score

from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

path = '/data1/mykim/predicting-satisfaction-using-graphs'

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # print(sum_mask)
    return sum_embeddings / sum_mask


# maximum sentence count (post + comment pair): 34
max_sentences = 34
embedding_size = 384


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


pe = PositionalEncoding(embedding_size, max_len=34)
x = torch.FloatTensor(max_sentences, 1, embedding_size).long()
x = pe(x)


class SplitBertModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.sbert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.fc_layer = nn.Linear(embedding_size, embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.pe = PositionalEncoding(embedding_size, max_len=max_sentences)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier1 = nn.Linear(embedding_size, embedding_size//2)
        self.classifier2 = nn.Linear(embedding_size//2, config.num_labels)
        self.classifier3 = nn.Linear(embedding_size, config.num_labels)
        self.post_init()

    def forward(
            self,
            post_input_ids: Optional[torch.tensor] = None,
            comment_input_ids: Optional[torch.tensor] = None,
            post_attention_mask: Optional[torch.tensor] = None,
            comment_attention_mask: Optional[torch.tensor] = None,
            post_sentence_count: Optional[torch.tensor] = None,
            comment_sentence_count: Optional[torch.tensor] = None,
            labels: Optional[torch.tensor] = None,
            indexes: Optional[torch.tensor] = None
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.SequenceClassifierOutput]:

        current_batch_size = len(post_input_ids)
        batch_encoder_embeddings = torch.empty(size=(current_batch_size, 1, max_post, embedding_size),
                                               requires_grad=True).to('cuda')
        batch_decoder_embeddings = torch.empty(size=(current_batch_size, 1, max_comment, embedding_size),
                                               requires_grad=True).to('cuda')

        def get_batch_embeddings(input_ids, attention_mask, sentence_count, max_sentences, batch_embeddings):
            for i in range(len(input_ids)):
                sbert_outputs = torch.empty(size=(max_sentences, 1, embedding_size)).to('cuda')

                model_output = self.sbert(input_ids[i][0:sentence_count[i]],
                                          attention_mask=attention_mask[i][0:sentence_count[i]])

                model_output = mean_pooling(model_output, attention_mask[i][0:sentence_count[i]])

                for j in range(sentence_count[i].item()):
                    sbert_outputs[j] = model_output[j]

                for j in range(sentence_count[i].item(), max_sentences):
                    sbert_outputs[j] = torch.zeros(1, embedding_size).to('cuda')

                # nn.Linear
                sbert_outputs_fc = self.fc_layer(sbert_outputs)
                sbert_outputs_fc = sbert_outputs_fc.swapaxes(0, 1)
                batch_embeddings[i] = sbert_outputs_fc

            return batch_embeddings

        batch_encoder_embeddings = get_batch_embeddings(post_input_ids, post_attention_mask,
                                                        post_sentence_count, max_post, batch_encoder_embeddings)
        batch_decoder_embeddings = get_batch_embeddings(comment_input_ids, comment_attention_mask,
                                                        comment_sentence_count, max_comment, batch_decoder_embeddings)

        def make_masks(max_sentences, count):
            mask = (torch.triu(torch.ones(max_sentences, max_sentences)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda')

            zeros = torch.zeros(1, count.item()).to('cuda')
            ones = torch.ones(1, max_sentences - count.item()).to('cuda')
            key_padding_mask = torch.cat([zeros, ones], dim=1).type(torch.bool)

            return mask, key_padding_mask

        outputs = torch.empty(size=(current_batch_size, embedding_size), requires_grad=True).to('cuda')

        for encoder_embeddings, decoder_embeddings, p_count, c_count, i in zip(batch_encoder_embeddings,
                                                                              batch_decoder_embeddings,
                                                                              post_sentence_count,
                                                                              comment_sentence_count,
                                                                              range(current_batch_size)):
            encoder_embeddings = encoder_embeddings.swapaxes(0, 1)
            decoder_embeddings = decoder_embeddings.swapaxes(0, 1)

            # add positional encoding
            encoder_embeddings = self.pe(encoder_embeddings)

            # make masks
            src_mask, src_key_padding_mask = make_masks(max_post, p_count)
            tgt_mask, tgt_key_padding_mask = make_masks(max_comment, c_count)

            encoder_output = self.encoder(encoder_embeddings,
                                          mask=src_mask,
                                          src_key_padding_mask=src_key_padding_mask)

            decoder_output = self.decoder(tgt=decoder_embeddings, memory=encoder_output,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask)

            decoder_output = torch.mean(decoder_output[:c_count], dim=0).squeeze(0)

            outputs[i] = decoder_output

        outputs = self.classifier1(outputs)
        logits = self.classifier2(outputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs
        )


nlp = English()
nlp.add_pipe("sentencizer")

# for linux
post_df = pd.read_csv(path + '/csv/dataset/liwc_post.csv', encoding='UTF-8')
comment_df = pd.read_csv(path + '/csv/dataset/liwc_comment.csv', encoding='UTF-8')

# for windows
# post_df = pd.read_csv('csv/liwc_post.csv', encoding='UTF-8')
# comment_df = pd.read_csv('csv/liwc_comment.csv', encoding='UTF-8')

# texts (x)
post_contents = list(post_df['content'])
comment_bodies = list(comment_df['content'])

post_sequences = []
comment_sequences = []

for post_content, comment_body in zip(post_contents, comment_bodies):
    post_sequences.append(list(map(lambda x: str(x), list(nlp(post_content).sents))))
    comment_sequences.append(list(map(lambda x: str(x), list(nlp(comment_body).sents))))

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

# print(satisfactions_float)
# print(satisfactions)

data = []

max_post = 0
max_comment = 0
i = 0
for post, comment, satisfaction, satisfaction_float in zip(post_sequences, comment_sequences,
                                                           satisfactions, satisfactions_float):
    if len(post) > max_post:
        max_post = len(post)
    if len(comment) > max_comment:
        max_comment = len(comment)
    data.append([i, post, comment, satisfaction, satisfaction_float])
    i += 1

# max_post_sentences: 29, max_comment_sentences: 10
print(max_post, max_comment)

columns = ['index', 'post_contents', 'comment_contents', 'label', 'score']
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

print(train_sample_df.post_contents.values[0])
print(train_sample_df.comment_contents.values[0])


def conduct_input_ids_and_attention_masks(str_values, label_values, score_values, index_values, max_sentences_list):
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
    return TensorDataset(pc_input_ids[0], pc_input_ids[1],
                         pc_attention_masks[0], pc_attention_masks[1],
                         pc_sentence_count[0], pc_sentence_count[1], labels, scores, indexes)


dataset_train = conduct_input_ids_and_attention_masks([train_sample_df.post_contents.values,
                                                       train_sample_df.comment_contents.values],
                                                      train_sample_df.label.values, train_sample_df.score.values,
                                                      train_sample_df.index.values, [max_post, max_comment])

dataset_val = conduct_input_ids_and_attention_masks([val_df.post_contents.values, val_df.comment_contents.values],
                                                    val_df.label.values, val_df.score.values, val_df.index.values,
                                                    [max_post, max_comment])


model = SplitBertModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', num_labels=len(labels))
for param in model.sbert.parameters():
    param.requires_grad = False

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

    embeddings, predictions, true_vals, true_scores, indexes = [], [], [], [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        one_hot_labels = torch.nn.functional.one_hot(batch[6], num_classes=len(labels))

        inputs = {'post_input_ids': batch[0],
                  'comment_input_ids': batch[1],
                  'post_attention_mask': batch[2],
                  'comment_attention_mask': batch[3],
                  'post_sentence_count': batch[4],
                  'comment_sentence_count': batch[5],
                  'labels': one_hot_labels.type(torch.float),
                  'indexes': batch[7]
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        hidden_states = outputs[2]
        loss_val_total += loss.item()

        hidden_states = hidden_states.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        label_ids = batch[6].cpu().numpy()
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

    accuracy_per_class(predictions, true_vals)

    return loss_val_avg, embeddings, predictions, true_vals, true_scores, indexes


model.to(device)

training_result = []

for epoch in tqdm(range(1, epochs + 1)):
    result_for_tsne = []
    model.train()
    loss_train_total = 0
    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    i = 0
    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        one_hot_labels = torch.nn.functional.one_hot(batch[6], num_classes=len(labels))

        inputs = {'post_input_ids': batch[0],
                  'comment_input_ids': batch[1],
                  'post_attention_mask': batch[2],
                  'comment_attention_mask': batch[3],
                  'post_sentence_count': batch[4],
                  'comment_sentence_count': batch[5],
                  'labels': one_hot_labels.type(torch.float),
                  'indexes': batch[7]
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
    val_loss, embeddings, predictions, true_vals, true_scores, indexes = evaluate(dataloader_validation)

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
