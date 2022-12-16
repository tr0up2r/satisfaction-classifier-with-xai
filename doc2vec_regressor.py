import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import csv
from sklearn.manifold import TSNE
import spacy
import textstat


nltk.download('punkt')

data_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/avg_satisfaction_raw_0-999.csv', encoding='ISO-8859-1')

list_satisfaction = list(data_df['satisfy_composite'])
list_tag = list(data_df['postIndex'])
list_post = list(data_df['postContent'])
list_comment = list(data_df['commentContent'])

list_content = []
for p, c in zip(list_post, list_comment):
    list_content.append(p+' '+c)


def train_doc2vec(data, tag, model_name, mode):
    tagged_data = [TaggedDocument(words=word_tokenize(term.lower()), tags=[tag[i]]) for i, term in enumerate(data)]

    max_epochs = 500
    vec_size = 100
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        if epoch % 100 == 0:
            print('iteration {0}'.format(epoch))

        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save(f"{model_name}")

    embeddings = []
    for i in range(1000):
        embeddings.append(list(model.dv[i]))

    if mode == "regression":
        return embeddings
    else:
        tsne = TSNE(random_state=42)
        data_tsne = tsne.fit_transform(embeddings)

        x_for_tsne = []
        y_for_tsne = []

        for xy in data_tsne:
            x_for_tsne.append(xy[0])
            y_for_tsne.append(xy[1])

        return x_for_tsne, y_for_tsne


embeddings = train_doc2vec(list_content, list_tag, "post_comment.model", "regression")

# features: sentence count, readability
data = []

nlp = spacy.load('en_core_web_sm')
for i in range(len(embeddings)):
    doc = nlp(list_content[i])
    sentence_tokens = [[token.text for token in sent] for sent in doc.sents]
    data.append([embeddings[i], list_satisfaction[i], len(sentence_tokens), textstat.flesch_reading_ease(list_comment[i]), list_tag[i]])
data = pd.DataFrame(data, columns=['x', 'y', 'c', 'r', 'i'])

# train-test split
test_size = 0.2
train, test = train_test_split(data, test_size=0.2)

x_train = torch.Tensor([x for x in list(train.x)])
y_train = torch.Tensor([[y] for y in list(train.y)])
c_train = torch.Tensor([[c] for c in list(train.c)])
r_train = torch.Tensor([[r] for r in list(train.r)])
i_train = torch.Tensor([[i] for i in list(train.i)])

x_test = torch.Tensor([x for x in list(test.x)])
y_test = torch.Tensor([[y] for y in list(test.y)])
c_test = torch.Tensor([[c] for c in list(test.c)])
r_test = torch.Tensor([[r] for r in list(test.r)])
i_test = torch.Tensor([[i] for i in list(test.i)])


train_data = TensorDataset(x_train, y_train, c_train, r_train, i_train)
test_data = TensorDataset(x_test, y_test, c_test, r_test, i_test)

batch_size = 5
train_dl = DataLoader(train_data, batch_size, shuffle=False)
test_dl = DataLoader(test_data, batch_size, shuffle=False)


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(52, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, x, count, readability):
        output = self.fc1(x)
        output = self.dropout(output)
        output = self.relu(output)
        logit = self.fc2(torch.cat((output, count, readability), dim=1))

        return logit


def regression(num_epochs, model, loss_fun, optimizer, train_dl, test_dl):
    training_result = []

    for epoch in range(1, num_epochs + 1):
        avg_loss = 0
        for xb, yb, cb, rb, ib in train_dl:
            pred = model(xb, cb, rb)
            loss = loss_fun(pred, yb)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, avg_loss / len(train_dl)))

        pred_df, true_df, val_loss, r2score = evaluate(test_dl)

        training_result.append([epoch, avg_loss / len(train_dl), val_loss, r2score])

        pred_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/doc2vec_features/pred/epoch_{epoch}_predicted_vals.csv')
        true_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/doc2vec_features/true/epoch_{epoch}_true_vals.csv')

    fields = ['epoch', 'training_loss', 'validation_loss', 'r^2_score']

    with open(
            f'../predicting-satisfaction-using-graphs/csv/doc2vec_features/training_result.csv',
            'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(training_result)


def evaluate(test_dl):
    avg_loss = 0
    predictions = []
    true_vals = []
    indexes = []
    for xb, yb, cb, rb, ib in test_dl:
        pred = model(xb, cb, rb)
        loss = loss_fun(pred, yb)
        avg_loss += loss.item()

        pred = pred.detach().cpu().numpy()
        yb = yb.detach().cpu().numpy()
        ib = ib.detach().cpu().numpy()

        predictions.append(pred)
        true_vals.append(yb)
        indexes.append(ib)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    indexes = np.concatenate(indexes, axis=0)

    pred_df = pd.DataFrame(torch.Tensor(predictions).tolist())
    true_df = pd.DataFrame(torch.Tensor(true_vals).tolist())
    index_df = pd.DataFrame(torch.Tensor(indexes).tolist())

    pred_df = pd.concat([pred_df, index_df], axis=1)
    true_df = pd.concat([true_df, index_df], axis=1)

    # print('validation loss: {:.4f}, R^2 score: {:.4f}'.format((avg_loss / len(test_dl)), r2_score(true_vals, predictions)))
    return pred_df, true_df, avg_loss / len(test_dl), r2_score(true_vals, predictions)


model = Regressor()
loss_fun = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

num_epochs = 7000
regression(num_epochs, model, loss_fun, optimizer, train_dl, test_dl)