import pandas as pd
import numpy as np
import random
import csv
import torch
from tqdm import tqdm
from utils import create_folder
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from evaluate import evaluate
from utils import f1_score_func


def train(model, device, dataset_train, dataset_val, labels, target, path, mode):
    if target == 'post_comment':
        mode_path = f"{mode[0]}_{mode[1]}"
    elif target == 'reply':
        mode_path = mode[0]
    else:
        mode_path = f"{mode[0]}_{mode[1]}_{mode[2]}"

    create_folder(path + f'/csv/splitbert_classifier/{target}/{mode_path}')
    create_folder(path + f'/splitbert/model/{mode_path}/')

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

    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    model.to(device)

    training_result = []

    # Training Loop
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        i = 0
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            one_hot_labels = torch.nn.functional.one_hot(batch[0], num_classes=len(labels))

            inputs = {'labels': one_hot_labels.type(torch.float),
                      'input_ids1': batch[1],
                      'input_ids2': batch[2],
                      'attention_mask1': batch[3],
                      'attention_mask2': batch[4],
                      'sentence_count1': batch[5],
                      'sentence_count2': batch[6]
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
        torch.save(model.state_dict(),
                   f'{path}/splitbert/model/{mode_path}/epoch_{epoch}.model')
        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        val_loss, predictions, true_vals, true_scores, indexes = evaluate(dataloader_validation, model,
                                                                                      device, target, labels, mode)
        preds_flat = np.argmax(predictions, axis=1).flatten()
        labels_flat = true_vals.flatten()
        scores_flat = true_scores.flatten()
        indexes_flat = indexes.flatten()
        # print(type(embeddings))

        val_f1_macro, val_f1_micro = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Macro, Micro): {val_f1_macro}, {val_f1_micro}')
        training_result.append([epoch, loss_train_avg, val_loss, val_f1_macro, val_f1_micro])

        tsne_df = pd.DataFrame({'prediction': preds_flat, 'label': labels_flat,
                                'score': scores_flat, 'index': indexes_flat})
        tsne_df.to_csv(path + f'/csv/splitbert_classifier/{target}/{mode_path}/epoch_{epoch}_result.csv')

    fields = ['epoch', 'training_loss', 'validation_loss', 'f1_score_macro', 'f1_score_micro']

    with open(
            path + f'/csv/splitbert_classifier/{target}/{mode_path}/training_result.csv',
            'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(training_result)