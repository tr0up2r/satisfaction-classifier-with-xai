import pandas as pd
import numpy as np
import random
import csv
import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertPreTrainedModel

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import modeling_outputs
from typing import Optional, Tuple, Union
from sklearn.metrics import f1_score

from spacy.lang.en import English
from transformers import AutoModel
from sklearn.metrics import r2_score


max_sentences = 34
max_post = 29
max_comment = 10
embedding_size = 384
num_labels = 3


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

        # self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()
        # self.classifier1 = nn.Linear(embedding_size, embedding_size//2)
        # self.classifier2 = nn.Linear(embedding_size//2, config.num_labels)
        # self.classifier3 = nn.Linear(embedding_size, config.num_labels)
        self.post_init()

    def forward(
            self,
            post_input_ids: Optional[torch.tensor] = None,
            comment_input_ids: Optional[torch.tensor] = None,
            post_attention_mask: Optional[torch.tensor] = None,
            comment_attention_mask: Optional[torch.tensor] = None,
            post_sentence_count: Optional[torch.tensor] = None,
            comment_sentence_count: Optional[torch.tensor] = None,
            prediction_mode: Optional[torch.tensor] = None,
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
                count = sentence_count[i].item()
                model_output = self.sbert(input_ids[i][0:count],
                                          attention_mask=attention_mask[i][0:count])

                model_output = mean_pooling(model_output, attention_mask[i][0:count])

                for j in range(count):
                    sbert_outputs[j] = model_output[j]

                for j in range(count, max_sentences):
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

        # results = self.classifier1(outputs)
        # logits = self.classifier2(results)
        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(logits.squeeze(), labels.squeeze())

        return outputs

        '''
         return modeling_outputs.SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs
        )
        '''


class ContrastiveLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.splitbert = SplitBertModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', num_labels=num_labels)
        self.regressor1 = nn.Linear(embedding_size*3, embedding_size)
        self.regressor2 = nn.Linear(embedding_size, 1)

    def forward(self,
                post_input_ids1, comment_input_ids1, post_input_ids2, comment_input_ids2,
                post_attention_mask1, comment_attention_mask1, post_attention_mask2, comment_attention_mask2,
                post_sentence_count1, comment_sentence_count1, post_sentence_count2, comment_sentence_count2,
                labels1, labels2, diffs, indexes1, indexes2):

        inputs1 = {'post_input_ids': post_input_ids1,
                   'comment_input_ids': comment_input_ids1,
                   'post_attention_mask': post_attention_mask1,
                   'comment_attention_mask': comment_attention_mask1,
                   'post_sentence_count': post_sentence_count1,
                   'comment_sentence_count': comment_sentence_count1,
                   'labels': labels1}

        inputs2 = {'post_input_ids': post_input_ids2,
                   'comment_input_ids': comment_input_ids2,
                   'post_attention_mask': post_attention_mask2,
                   'comment_attention_mask': comment_attention_mask2,
                   'post_sentence_count': post_sentence_count2,
                   'comment_sentence_count': comment_sentence_count2,
                   'labels': labels2}

        hidden_states1 = self.splitbert(**inputs1)
        hidden_states2 = self.splitbert(**inputs2)

        # hidden_states1 = outputs1['hidden_states']
        # hidden_states2 = outputs2['hidden_states']
        difference = torch.sub(hidden_states1, hidden_states2)

        # print(difference)

        outputs = torch.cat([hidden_states1, hidden_states2, difference], dim=1)
        outputs = self.regressor1(outputs)
        logits = self.regressor2(outputs)

        loss = None
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze().float(), diffs.squeeze().float())
        # print(loss.item())

        return (loss, logits)


def make_train_test_df(df):

    data = []
    columns = ['index1', 'post_contents1', 'comment_contents1', 'label1', 'label_for_contrastive1', 'score1',
               'index2', 'post_contents2', 'comment_contents2', 'label2', 'label_for_contrastive2', 'score2']

    index1 = list(df[columns[0]])
    post_contents1 = list(df[columns[1]])
    comment_contents1 = list(df[columns[2]])
    label1 = list(df[columns[3]])
    label_for_contrastive1 = list(df[columns[4]])
    score1 = list(df[columns[5]])

    index2 = list(df[columns[6]])
    post_contents2 = list(df[columns[7]])
    comment_contents2 = list(df[columns[8]])
    label2 = list(df[columns[9]])
    label_for_contrastive2 = list(df[columns[10]])
    score2 = list(df[columns[11]])

    post_sequences1 = []
    comment_sequences1 = []
    post_sequences2 = []
    comment_sequences2 = []

    for p1, c1, p2, c2 in zip(post_contents1, comment_contents1, post_contents2, comment_contents2):
        post_sequences1.append(list(map(lambda x: str(x), list(nlp(p1).sents))))
        comment_sequences1.append(list(map(lambda x: str(x), list(nlp(c1).sents))))
        post_sequences2.append(list(map(lambda x: str(x), list(nlp(p2).sents))))
        comment_sequences2.append(list(map(lambda x: str(x), list(nlp(c2).sents))))

    for i1, p1, c1, l1, lc1, s1, i2, p2, c2, l2, lc2, s2 in zip(index1, post_sequences1, comment_sequences1, label1,
                                                            label_for_contrastive1, score1, index2, post_sequences2,
                                                            comment_sequences2, label2, label_for_contrastive2, score2):

        data.append([i1, p1, c1, l1, lc1, i2, p2, c2, l2, lc2, s1-s2])

    columns = ['index1', 'post_contents1', 'comment_contents1', 'label1', 'label_for_contrastive1',
               'index2', 'post_contents2', 'comment_contents2', 'label2', 'label_for_contrastive2', 'difference']

    df = pd.DataFrame(data, columns=columns)

    return df


def conduct_input_ids_and_attention_masks(str_values, label_values1, label_values2, diff_values,
                                          index_values1, index_values2, max_sentences_list):
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

    labels1 = torch.tensor(label_values1.astype(int))
    labels2 = torch.tensor(label_values2.astype(int))
    diffs = torch.tensor(diff_values.astype(float))
    indexes1 = torch.tensor(index_values1.astype(int))
    indexes2 = torch.tensor(index_values2.astype(int))

    # 0: posts / 1: comments
    return TensorDataset(pc_input_ids[0], pc_input_ids[1], pc_input_ids[2], pc_input_ids[3],
                         pc_attention_masks[0], pc_attention_masks[1], pc_attention_masks[2], pc_attention_masks[3],
                         pc_sentence_count[0], pc_sentence_count[1], pc_sentence_count[2], pc_sentence_count[3],
                         labels1, labels2, diffs, indexes1, indexes2)


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


def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0

    embeddings, predictions, true_vals, true_scores, indexes = [], [], [], [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        one_hot_labels1 = torch.nn.functional.one_hot(batch[12], num_classes=num_labels)
        one_hot_labels2 = torch.nn.functional.one_hot(batch[13], num_classes=num_labels)

        inputs = {'post_input_ids1': batch[0],
                  'comment_input_ids1': batch[1],
                  'post_input_ids2': batch[2],
                  'comment_input_ids2': batch[3],
                  'post_attention_mask1': batch[4],
                  'comment_attention_mask1': batch[5],
                  'post_attention_mask2': batch[6],
                  'comment_attention_mask2': batch[7],
                  'post_sentence_count1': batch[8],
                  'comment_sentence_count1': batch[9],
                  'post_sentence_count2': batch[10],
                  'comment_sentence_count2': batch[11],
                  'labels1': one_hot_labels1.type(torch.float),
                  'labels2': one_hot_labels2.type(torch.float),
                  'diffs': batch[14],
                  'indexes1': batch[15],
                  'indexes2': batch[16]
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        diff_ids = inputs['diffs'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(diff_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    # embeddings = np.concatenate(embeddings, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    # true_scores = np.concatenate(true_scores, axis=0)
    # indexes = np.concatenate(indexes, axis=0)

    # accuracy_per_class(predictions, true_vals)

    return loss_val_avg, predictions, true_vals


if __name__ == '__main__':
    print('Running!')
    # maximum sentence count (post + comment pair): 34

    pe = PositionalEncoding(embedding_size, max_len=34)
    x = torch.FloatTensor(max_sentences, 1, embedding_size).long()
    x = pe(x)

    # prepare dataset
    nlp = English()
    nlp.add_pipe("sentencizer")

    # for linux
    df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/df_for_contrastive_learner_train_50.csv',
                     encoding='UTF-8')
    test_df = pd.read_csv(
        '../predicting-satisfaction-using-graphs/csv/dataset/df_for_contrastive_learner_test_remain.csv',
        encoding='UTF-8')

    # for windows
    # df = pd.read_csv('csv/df_for_contrastive_learner_mini.csv', encoding='UTF-8')
    print(df.columns)
    print(test_df.columns)

    '''
    # data split (train & test sets)
    idx_train, idx_val = train_test_split(df.index.values, test_size=0.20, random_state=42)

    train_df = df.iloc[idx_train]
    val_df = df.iloc[idx_val]
    '''

    train_df = make_train_test_df(df)
    val_df = make_train_test_df(test_df)

    print(train_df.shape)
    print(val_df.shape)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)

    dataset_train = conduct_input_ids_and_attention_masks([train_df.post_contents1.values,
                                                           train_df.comment_contents1.values,
                                                           train_df.post_contents2.values,
                                                           train_df.comment_contents2.values, ],
                                                          train_df.label1.values, train_df.label2.values,
                                                          train_df.difference.values, train_df.index1.values,
                                                          train_df.index2.values, [max_post, max_comment] * 2)

    dataset_val = conduct_input_ids_and_attention_masks([val_df.post_contents1.values, val_df.comment_contents1.values,
                                                         val_df.post_contents2.values,
                                                         val_df.comment_contents2.values, ],
                                                        val_df.label1.values, val_df.label2.values,
                                                        val_df.difference.values, val_df.index1.values,
                                                        val_df.index2.values, [max_post, max_comment] * 2)

    # model = SplitBertModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', num_labels=num_labels)
    model = ContrastiveLearner()
    for param in model.splitbert.sbert.parameters():
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

    # Training Loop
    device = torch.device('cuda')
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

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
            one_hot_labels1 = torch.nn.functional.one_hot(batch[12], num_classes=num_labels)
            one_hot_labels2 = torch.nn.functional.one_hot(batch[13], num_classes=num_labels)

            inputs = {'post_input_ids1': batch[0],
                      'comment_input_ids1': batch[1],
                      'post_input_ids2': batch[2],
                      'comment_input_ids2': batch[3],
                      'post_attention_mask1': batch[4],
                      'comment_attention_mask1': batch[5],
                      'post_attention_mask2': batch[6],
                      'comment_attention_mask2': batch[7],
                      'post_sentence_count1': batch[8],
                      'comment_sentence_count1': batch[9],
                      'post_sentence_count2': batch[10],
                      'comment_sentence_count2': batch[11],
                      'labels1': one_hot_labels1.type(torch.float),
                      'labels2': one_hot_labels2.type(torch.float),
                      'diffs': batch[14],
                      'indexes1': batch[15],
                      'indexes2': batch[16]
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
        val_loss, predictions, true_vals = evaluate(dataloader_validation)
        preds_flat = predictions.flatten()
        # print(preds_flat)
        labels_flat = true_vals.flatten()
        # print(labels_flat)
        # print(type(embeddings))

        # val_f1_macro, val_f1_micro = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'R^2 Score: {r2_score(labels_flat, preds_flat)}')
        training_result.append([epoch, loss_train_avg, val_loss, r2_score(labels_flat, preds_flat)])

        tsne_df = pd.DataFrame({'prediction': preds_flat, 'label': labels_flat})
        tsne_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/contrastive_learner/epoch_{epoch}_result.csv')
        print(model.splitbert.state_dict())
        torch.save(model.splitbert.state_dict(),
                   f'../predicting-satisfaction-using-graphs/model/contrastive_learner/epoch_{epoch}_model.pt')

    fields = ['epoch', 'training_loss', 'validation_loss', 'r^2_score']

    with open(
            f'../predicting-satisfaction-using-graphs/csv/contrastive_learner/training_result.csv',
            'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(training_result)