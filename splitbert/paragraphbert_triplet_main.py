from itertools import product
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
    # [post_mode, comment_mode, reply_mode]
    items = ['all', 'seg', 'snt']
    modes = list(map(lambda x: list(x), list(product(items, items, items))))
    
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

    reply_contents = list(reply_df['replyContent'])
    post_contents = list(post_df['content'])
    comment_bodies = list(comment_df['content'])

    def get_sequences(contents, mode):
        sequences = []

        if mode == 'all':
            for content in contents:
                sequences.append([content])
        elif mode == 'seg':
            for content in contents:
                sentences = list(map(lambda x: str(x), list(nlp(content).sents)))
                sequences.append(text_segmentation(sentences))
        else:  # sentences
            for content in contents:
                sequences.append(list(map(lambda x: str(x), list(nlp(content).sents))))

        return sequences

    for mode in modes:
        print(mode)
        post_sequences = get_sequences(post_contents, mode[0])
        comment_sequences = get_sequences(comment_bodies, mode[1])
        reply_sequences = get_sequences(reply_contents, mode[2])

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

        model = SplitBertConcatEncoderModel(num_labels=len(labels), embedding_size=384, max_len=max_count,
                                            pc_segmentation=False)

        device = torch.device('cuda')
        model.to(device)

        for param in model.sbert.parameters():
            param.requires_grad = False

        for param in model.bert.parameters():
            param.requires_grad = False

        train(model, device, dataset_train, dataset_val, labels, 'triplet', path, mode)
