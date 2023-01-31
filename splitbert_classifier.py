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


# maximum sentence count (post + comment pair): 34
max_sentences = 34


class SplitBertModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.embedding = nn.Embedding(768, 1)
        self.fc_layer = nn.Linear(768, 768)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier1 = nn.Linear(768, 768//2)
        self.classifier2 = nn.Linear(768//2, config.num_labels)
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.tensor] = None,
            attention_mask: Optional[torch.tensor] = None,
            sentence_count: Optional[torch.tensor] = None,
            labels: Optional[torch.tensor] = None
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.SequenceClassifierOutput]:

        sent_count_list = []
        batch_embeddings = torch.empty(size=(len(input_ids), 1, max_sentences, 768), requires_grad=True).to('cuda')
        # print(batch_embeddings)

        # print(self.classifier.requires_grad_())
        for i in range(len(input_ids)):

            bert_outputs = torch.empty(size=(max_sentences, 1, 768)).to('cuda')

            '''
            # print(input_ids.shape)
            # print(input_ids[i][j].shape)
            outputs = self.bert(
                # torch.tensor([input_ids[i][j].tolist()]).to('cuda'),
                # attention_mask=torch.tensor([attention_mask[i][j].tolist()]).to('cuda')
                input_ids[i],
                attention_mask=attention_mask[i]
            )
            # batch_embeddings.append(torch.tensor([outputs['pooler_output'].tolist()]).to('cuda'))
            # print(outputs['pooler_output'].unsqueeze(0).shape)
            # print(batch_embeddings[i].shape, outputs['pooler_output'].unsqueeze(0))
            batch_embeddings[i] = outputs['pooler_output'].unsqueeze(0)
            '''

            # print(batch_embeddings.shape)

            # print(batch_embeddings)

            for j in range(sentence_count[i].item()):
                outputs = self.bert(
                    input_ids[i][j].unsqueeze(0),
                    attention_mask=attention_mask[i][j].unsqueeze(0)
                )
                bert_outputs[j] = outputs['pooler_output']
                # print(self.fc_layer(bert_outputs[j]).shape)
                # bert_outputs[j] = self.fc_layer(bert_outputs[j])
                bert_outputs[j] = bert_outputs[j]
                # print(bert_outputs[j])
                # print(bert_outputs[j].shape)

            for j in range(sentence_count[i].item(), max_sentences):
                bert_outputs[j] = torch.zeros(1, 768).to('cuda')

            # print(bert_outputs)
            # print(bert_outputs)
            # print(self.fc_layer(bert_outputs))
            # print(bert_outputs.shape)

            # print(self.embedding.weight.shape)

            # bert_outputs_embedding = self.embedding(bert_outputs.swapaxes(0, 1).squeeze().long())
            # bert_outputs_embedding = bert_outputs_embedding.swapaxes(1, 2).swapaxes(0, 1)
            # print(bert_outputs_embedding.shape)
            # print(bert_outputs.swapaxes(0, 1).squeeze().shape)
            bert_outputs_fc = self.fc_layer(bert_outputs)
            bert_outputs_fc = bert_outputs_fc.swapaxes(0, 1)
            batch_embeddings[i] = bert_outputs_fc
            # batch_embeddings[i] = bert_outputs_embedding



        '''
        for embeddings, sent_count in zip(batch_embeddings, sent_count_list):
            # need to be with mask
            embeddings = embeddings.swapaxes(0, 1)
            # ones = torch.ones(max_sentences, sent_count).to('cuda')
            # zeros = torch.zeros(max_sentences, max_sentences-sent_count).to('cuda')

            zeros = torch.zeros(1, sent_count).to('cuda')
            ones = torch.ones(1, max_sentences-sent_count).to('cuda')
            src_key_padding_mask = torch.cat([zeros, ones], dim=1).type(torch.bool)

            encoder_output = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)[0][0]
            # encoder_output = encoder_output.masked_fill(torch.isnan(encoder_output), 0)
            encoder_outputs.append(encoder_output)
        '''


        # print(batch_embeddings.shape)

        def make_src_mask(sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda')
            return mask

        encoder_outputs = torch.empty(size=(len(input_ids), 768), requires_grad=True).to('cuda')

        for embeddings, count, i in zip(batch_embeddings, sentence_count, range(len(input_ids))):
            # ones = torch.ones(max_sentences, count.item()).to('cuda')
            # zeros = torch.zeros(max_sentences, max_sentences - count.item()).to('cuda')
            # src_mask = torch.cat([ones, zeros], dim=1).to('cuda')
            # src_mask = torch.zeros(max_sentences, max_sentences).to('cuda').type(torch.bool)
            embeddings = embeddings.swapaxes(0, 1)

            src_mask = make_src_mask(max_sentences)

            # print(src_mask)

            zeros = torch.zeros(1, count.item()).to('cuda')
            ones = torch.ones(1, max_sentences - count.item()).to('cuda')
            src_key_padding_mask = torch.cat([zeros, ones], dim=1).type(torch.bool)

            # print(embeddings.shape)
            # print(src_mask.shape)
            # print(src_key_padding_mask.shape)

            # print(embeddings)
            encoder_output = self.encoder(embeddings,
                                          mask=src_mask,
                                          src_key_padding_mask=src_key_padding_mask)[0][0]

            encoder_output_print = self.encoder(embeddings,
                                          mask=src_mask,
                                          src_key_padding_mask=src_key_padding_mask)

            for output in encoder_output_print:
                print(output)
            # print(encoder_output_print.shape)


            encoder_outputs[i] = encoder_output
            # print(encoder_output.shape)

        # print(encoder_outputs.shape)

        # encoder_outputs = torch.stack(encoder_outputs, dim=0).to('cuda')
        # print(encoder_outputs.shape)
        encoder_outputs = self.classifier1(encoder_outputs)
        encoder_outputs = self.dropout(encoder_outputs)
        encoder_outputs = self.relu(encoder_outputs)
        logits = self.classifier2(encoder_outputs)
        # print(logits)
        loss_fct = nn.CrossEntropyLoss()
        # loss_fct = nn.NLLLoss()
        # loss_fct = nn.BCEWithLogitsLoss(reduction='sum')

        loss = loss_fct(logits.squeeze(), labels.squeeze())
        # loss = loss_fct(nn.functional.log_softmax(logits, dim=1).squeeze(), labels.squeeze())

        # print(loss)
        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


nlp = English()
nlp.add_pipe("sentencizer")

# for linux
post_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/liwc_post.csv', encoding='UTF-8')
comment_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/liwc_comment.csv', encoding='UTF-8')

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

data = []

for post, comment, satisfaction in zip(post_sequences, comment_sequences, satisfactions):
    data.append([post+comment, satisfaction])

columns = ['contents', 'label']
df = pd.DataFrame(data, columns=columns)

# data split (train & test sets)
idx_train, idx_remain = train_test_split(df.index.values, test_size=0.20, random_state=42)
idx_val, idx_test = train_test_split(idx_remain, test_size=0.50, random_state=42)

# print(idx_train.shape, idx_val.shape, idx_test.shape)

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

# print(train_sample_df.contents.values[0])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)

# 이렇게 하면 한 document에 대해 input_ids가 문장 단위로 쪼개져서 나옴.
'''
result = tokenizer.batch_encode_plus(
    train_sample_df.contents.values[0],
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512,
    return_tensors='pt'
)

'''


def conduct_input_ids_and_attention_masks(str_values, score_values):
    input_ids_list = []
    attention_masks_list = []
    sentence_count_list = []

    # print(str_values)

    for contents in str_values:
        result = tokenizer.batch_encode_plus(
            contents,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=512,
            return_tensors='pt'
        )

        input_ids = result['input_ids']
        attention_masks = result['attention_mask']

        sentence_count_list.append(len(input_ids))

        # add zero pads to make all tensors' dimension (34, 512)
        pad = (0, 0, 0, max_sentences-len(input_ids))
        input_ids = nn.functional.pad(input_ids, pad, "constant", 0)  # effectively zero padding
        attention_masks = nn.functional.pad(attention_masks, pad, "constant", 0)

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_masks)

    input_ids = torch.stack(input_ids_list, dim=0)
    attention_masks = torch.stack(attention_masks_list, dim=0)
    sentence_counts = torch.tensor(sentence_count_list)
    labels = torch.tensor(score_values.astype(int))

    print(input_ids.shape, attention_masks.shape, labels.shape, sentence_counts.shape)

    return TensorDataset(input_ids, attention_masks, sentence_counts, labels)


dataset_train = conduct_input_ids_and_attention_masks(train_sample_df.contents.values,
                                                      train_sample_df.label.values)

dataset_val = conduct_input_ids_and_attention_masks(val_df.contents.values,
                                                    val_df.label.values)


model = SplitBertModel.from_pretrained('bert-base-uncased', num_labels=len(labels))
for param in model.bert.parameters():
    param.requires_grad = False

optimizer = AdamW(model.parameters(),
                  lr=2e-4,
                  eps=1e-8)

epochs = 10
# print(model.parameters())

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

    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        one_hot_labels = torch.nn.functional.one_hot(batch[3], num_classes=len(labels))

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'sentence_count': batch[2],
                  'labels': one_hot_labels.type(torch.float)
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        logits = logits.detach().cpu().numpy()

        label_ids = batch[3].cpu().numpy()

        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    print(predictions)
    true_vals = np.concatenate(true_vals, axis=0)
    print(true_vals)

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
        batch = tuple(b.to(device) for b in batch)
        one_hot_labels = torch.nn.functional.one_hot(batch[3], num_classes=len(labels))

        # print(one_hot_labels.type(torch.float))
        # print(batch[2])

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'sentence_count': batch[2],
                  'labels': one_hot_labels.type(torch.float),
                  }

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
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1_macro, val_f1_micro = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Macro, Micro): {val_f1_macro}, {val_f1_micro}')
    training_result.append([epoch, loss_train_avg, val_loss, val_f1_macro, val_f1_micro])

fields = ['epoch', 'training_loss', 'validation_loss', 'f1_score_macro', 'f1_score_micro']

with open(
        f'../predicting-satisfaction-using-graphs/csv/splitbert_classifier/training_result.csv',
        'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(training_result)