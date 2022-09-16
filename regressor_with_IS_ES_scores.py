import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import r2_score
import csv


data_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/seeker_satisfaction_1000_thread.csv', encoding='ISO-8859-1')

post_IS = list(data_df['post_IS'])
post_ES = list(data_df['post_ES'])
comment_IS = list(data_df['comment_IS'])
comment_ES = list(data_df['comment_ES'])
satisfaction = list(data_df['satisfaction'])

data = []
for i in range(len(post_IS)):
    data.append([[post_IS[i], post_ES[i], comment_IS[i], comment_ES[i]], satisfaction[i]])
data = pd.DataFrame(data, columns=['x', 'y'])

# train-test split
test_size = 0.2
train, test = train_test_split(data, test_size=0.2)

x_train = torch.Tensor([x for x in list(train.x)])
y_train = torch.Tensor([[y] for y in list(train.y)])

x_test = torch.Tensor([x for x in list(test.x)])
y_test = torch.Tensor([[y] for y in list(test.y)])


train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

batch_size = 5
train_dl = DataLoader(train_data, batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size, shuffle=True)


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(4, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        logit = self.fc2(self.fc1(x))

        return logit


model = Regressor()
loss_fun = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


def regression(num_epochs, model, loss_fun, optimizer, train_dl, test_dl):
    training_result = []

    for epoch in range(1, num_epochs+1):
        avg_loss = 0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fun(pred, yb)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, avg_loss / len(train_dl)))

        pred_df, true_df, val_loss, r2score = evaluate(test_dl)

        training_result.append([epoch, avg_loss / len(train_dl), val_loss, r2score])

        pred_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/fcregressor/epoch_{epoch}_predicted_vals.csv')

        # validation_loss, r2 = evaluate(test_dl)
    true_df.to_csv(f'../predicting-satisfaction-using-graphs/csv/fcregressor/true_vals.csv')

    fields = ['epoch', 'training_loss', 'validation_loss', 'r^2_score']

    with open(
            f'../predicting-satisfaction-using-graphs/csv/fcregressor/training_result.csv',
            'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(training_result)


def evaluate(test_dl):
    avg_loss = 0
    predictions = []
    true_vals = []
    for xb, yb in test_dl:
        pred = model(xb)
        loss = loss_fun(pred, yb)
        avg_loss += loss.item()

        pred = pred.detach().cpu().numpy()
        yb = yb.detach().cpu().numpy()

        predictions.append(pred)
        true_vals.append(yb)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    pred_df = pd.DataFrame(torch.Tensor(predictions).tolist())
    true_df = pd.DataFrame(torch.Tensor(true_vals).tolist())

    # print('validation loss: {:.4f}, R^2 score: {:.4f}'.format((avg_loss / len(test_dl)), r2_score(true_vals, predictions)))
    return pred_df, true_df, avg_loss / len(test_dl), r2_score(true_vals, predictions)


num_epochs = 7000
regression(num_epochs, model, loss_fun, optimizer, train_dl, test_dl)