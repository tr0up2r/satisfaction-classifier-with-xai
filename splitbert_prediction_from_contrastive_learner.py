import pandas as pd
import torch
import torch.nn as nn

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from spacy.lang.en import English
from contrastive_learning_with_splitberts_testset import SplitBertModel

device = torch.device('cuda')
num_labels = 1
embedding_size = 384


class SplitbertPredictiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.splitbert = SplitBertModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', num_labels=num_labels)
        # self.fc_layer1 = nn.Linear(embedding_size, embedding_size//2)
        self.fc_layer1 = nn.Linear(embedding_size, num_labels)
        self.fc_layer2 = nn.Linear(embedding_size//2, num_labels)

    def forward(self,
                post_input_ids, comment_input_ids, post_attention_mask, comment_attention_mask,
                post_sentence_count, comment_sentence_count):

        self.splitbert.load_state_dict(
            torch.load(
                '../predicting-satisfaction-using-graphs/model/contrastive_learner/classification/epoch_10_model.pt'))
        self.splitbert.to(device)

        inputs = {'post_input_ids': post_input_ids,
                  'comment_input_ids': comment_input_ids,
                  'post_attention_mask': post_attention_mask,
                  'comment_attention_mask': comment_attention_mask,
                  'post_sentence_count': post_sentence_count,
                  'comment_sentence_count': comment_sentence_count,
                  'prediction_mode': True}

        outputs = self.splitbert(**inputs)
        logits = self.fc_layer1(outputs)
        # logits = self.fc_layer2(outputs)

        '''
        loss = None
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze().float(), diffs.squeeze().float())
        # print(loss.item())
        '''

        return logits
        return (loss, logits)


nlp = English()
nlp.add_pipe("sentencizer")

# for linux
post_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/liwc_post.csv', encoding='UTF-8')
comment_df = pd.read_csv('../predicting-satisfaction-using-graphs/csv/dataset/liwc_comment.csv', encoding='UTF-8')

print(post_df.shape)

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

max_post = 0
max_comment = 0
i = 0

print(len(post_sequences), len(comment_sequences), len(satisfactions), len(satisfactions_float))

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

print(df.shape)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)


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

dataset = conduct_input_ids_and_attention_masks([df.post_contents.values,
                                                 df.comment_contents.values],
                                                df.label.values, df.score.values,
                                                df.index.values, [max_post, max_comment])

model = SplitbertPredictiveModel()
model.to(device)
model.eval()

with torch.no_grad():
    for batch in dataset:
        batch = list(batch)
        for i in range(len(batch)):
            batch[i] = batch[i].unsqueeze(0).to(device)

        inputs = {'post_input_ids': batch[0],
                  'comment_input_ids': batch[1],
                  'post_attention_mask': batch[2],
                  'comment_attention_mask': batch[3],
                  'post_sentence_count': batch[4],
                  'comment_sentence_count': batch[5]
                  }

        print(model(**inputs))