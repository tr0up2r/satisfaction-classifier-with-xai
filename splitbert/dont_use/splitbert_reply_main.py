import pandas as pd
from spacy.lang.en import English
from transformers import BertTokenizer
import torch

from splitbert import train_test_split
from splitbert import conduct_input_ids_and_attention_masks
from splitbert import SplitBertEncoderModel
from splitbert import train


if __name__ == "__main__":
    path = '/data1/mykim/predicting-satisfaction-using-graphs'
    nlp = English()
    nlp.add_pipe("sentencizer")

    # for linux
    reply_df = pd.read_csv(path + '/csv/dataset/avg_satisfaction_raw_0-999.csv', encoding='ISO-8859-1')

    # texts (x)
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

    reply_sequences = []

    for reply in reply_contents:
        reply_sequences.append(list(map(lambda x: str(x), list(nlp(reply).sents))))

    data = []
    max_reply = 0
    i = 0
    for reply, satisfaction, satisfaction_float in zip(reply_sequences, satisfactions, satisfactions_float):
        if len(reply) > max_reply:
            max_reply = len(reply)
        data.append([i, reply, satisfaction, satisfaction_float])
        i += 1

    # max_reply: 10
    print(max_reply)
    print(sum(map(len, reply_sequences)) / len(reply_sequences))
    exit()

    columns = ['index', 'reply_contents', 'label', 'score']
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

    print(train_sample_df.reply_contents.values[0])
    print(train_sample_df.shape)

    dataset_train = conduct_input_ids_and_attention_masks(tokenizer, [train_sample_df.reply_contents.values],
                                                          train_sample_df.label.values, train_sample_df.score.values,
                                                          train_sample_df.index.values, [max_reply], 'reply')

    dataset_val = conduct_input_ids_and_attention_masks(tokenizer, [val_df.reply_contents.values],
                                                        val_df.label.values, val_df.score.values,
                                                        val_df.index.values, [max_reply], 'reply')

    model = SplitBertEncoderModel(num_labels=len(labels), embedding_size=384, max_len=max_reply)

    device = torch.device('cuda')
    model.to(device)

    for param in model.sbert.parameters():
        param.requires_grad = False

    train(model, device, dataset_train, dataset_val, labels, 'reply', path)
