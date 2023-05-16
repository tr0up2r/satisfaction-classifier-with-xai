import pandas as pd
from spacy.lang.en import English
from collections import Counter
from transformers import BertTokenizer
import torch

from splitbert import train_test_split
from splitbert import conduct_input_ids_and_attention_masks
from splitbert import SplitBertConcatEncoderModel
from splitbert import SplitBertEncoderModel
from splitbert import train
from textsplit import text_segmentation


if __name__ == "__main__":
    path = '/data1/mykim/predicting-satisfaction-using-graphs'
    post_mode = 'all'
    comment_mode = 'all'
    reply_mode = 'sentence'  # sentence or segmentation
    nlp = English()
    nlp.add_pipe("sentencizer")

    # for linux
    post_df = pd.read_csv(path + '/csv/dataset/liwc_post.csv', encoding='UTF-8')
    comment_df = pd.read_csv(path + '/csv/dataset/liwc_comment.csv', encoding='UTF-8')
    reply_df = pd.read_csv(path + '/csv/dataset/avg_satisfaction_raw_0-999.csv', encoding='ISO-8859-1')

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

    post_sequences = []
    comment_sequences = []
    reply_sequences = []
    # print(post_contents[0])

    # texts (x)

    reply_contents = list(reply_df['replyContent'])
    post_contents = list(post_df['content'])
    comment_bodies = list(comment_df['content'])

    if post_mode == 'all' and comment_mode == 'all':
        for post_content, comment_body in zip(post_contents, comment_bodies):
            post_sequences.append([post_content])
            comment_sequences.append([comment_body])
    else:
        post_contents = list(post_df['content'])
        comment_bodies = list(comment_df['content'])
        for post_content, comment_body in zip(post_contents, comment_bodies):
            post_sentences = list(map(lambda x: str(x), list(nlp(post_content).sents)))
            post_sequences.append(text_segmentation(post_sentences))

            comment_sentences = list(map(lambda x: str(x), list(nlp(comment_body).sents)))
            comment_sequences.append(text_segmentation(comment_sentences))

    print(len(post_sequences[0]))

    if reply_mode == 'sentence':
        for reply in reply_contents:
            reply_sequences.append(list(map(lambda x: str(x), list(nlp(reply).sents))))
    else:
        for reply in reply_contents:
            reply_sentences = list(map(lambda x: str(x), list(nlp(reply).sents)))
            reply_sequences.append(text_segmentation(reply_sentences))

    data = []
    max_post, max_comment, max_reply = 0, 0, 0
    i = 0
    for post, comment, reply, satisfaction, satisfaction_float in zip(post_sequences, comment_sequences,
                                                                      reply_sequences, satisfactions,
                                                                      satisfactions_float):
        if len(post) > max_post:
            max_post = len(post)
        if len(comment) > max_comment:
            max_comment = len(comment)
        if len(reply) > max_reply:
            max_reply = len(reply)

        data.append([i, post, comment, reply, satisfaction, satisfaction_float])
        i += 1

    # max post / comment / reply: 10 / 4 / 4
    print(max_post, max_comment, max_reply)
    max_count = max(max_post, max_comment, max_reply)
    print(max_count)

    columns = ['index', 'post_contents', 'comment_contents', 'reply_contents', 'label', 'score']
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

    dataset_train = conduct_input_ids_and_attention_masks(tokenizer, [train_sample_df.post_contents.values,
                                                                      train_sample_df.comment_contents.values,
                                                                      train_sample_df.reply_contents.values],
                                                          train_sample_df.label.values, train_sample_df.score.values,
                                                          train_sample_df.index.values, max_count, 'triplet')

    dataset_val = conduct_input_ids_and_attention_masks(tokenizer, [val_df.post_contents.values,
                                                                    val_df.comment_contents.values,
                                                                    val_df.reply_contents.values],
                                                        val_df.label.values, val_df.score.values,
                                                        val_df.index.values, max_count, 'triplet')

    print(dataset_val[0])

    model = SplitBertConcatEncoderModel(num_labels=len(labels), embedding_size=384, max_len=max_count,
                                        pc_segmentation=False)

    device = torch.device('cuda')
    model.to(device)

    for param in model.sbert.parameters():
        param.requires_grad = False

    train(model, device, dataset_train, dataset_val, labels, 'triplet', path)
