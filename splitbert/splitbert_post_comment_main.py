import pandas as pd
from spacy.lang.en import English
from transformers import BertTokenizer
import torch

from splitbert import train_test_split
from splitbert import conduct_input_ids_and_attention_masks
from splitbert import SplitBertTransformerModel
from splitbert import train


if __name__ == "__main__":
    path = '/data1/mykim/predicting-satisfaction-using-graphs'
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

    dataset_train = conduct_input_ids_and_attention_masks(tokenizer, [train_sample_df.post_contents.values,
                                                                      train_sample_df.comment_contents.values],
                                                          train_sample_df.label.values, train_sample_df.score.values,
                                                          train_sample_df.index.values, [max_post, max_comment],
                                                          'post_comment')

    dataset_val = conduct_input_ids_and_attention_masks(tokenizer, [val_df.post_contents.values,
                                                                    val_df.comment_contents.values],
                                                        val_df.label.values, val_df.score.values,
                                                        val_df.index.values, [max_post, max_comment],
                                                        'post_comment')

    model = SplitBertTransformerModel(num_labels=len(labels), embedding_size=384, max_sentences=34,
                                      max_len1=max_post, max_len2=max_comment)

    device = torch.device('cuda')
    model.to(device)

    for param in model.sbert.parameters():
        param.requires_grad = False

    train(model, device, dataset_train, dataset_val, labels, 'post_comment', path)
