import numpy as np
import torch
from utils import accuracy_per_class


def evaluate(dataloader_val, model, device, target, labels, mode):
    model.eval()
    loss_val_total = 0

    embeddings, predictions, true_vals, true_scores, indexes = [], [], [], [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        one_hot_labels = torch.nn.functional.one_hot(batch[0], num_classes=len(labels))

        if target == 'post_comment':
            inputs = {'labels': one_hot_labels.type(torch.float),
                      'input_ids1': batch[1],
                      'input_ids2': batch[2],
                      'attention_mask1': batch[3],
                      'attention_mask2': batch[4],
                      'sentence_count1': batch[5],
                      'sentence_count2': batch[6]
                      }
        elif target == 'reply':
            inputs = {'labels': one_hot_labels.type(torch.float),
                      'input_ids1': batch[1],
                      'attention_mask1': batch[2],
                      'sentence_count1': batch[3],
                      'mode': mode
                      }
        else:
            inputs = {'labels': one_hot_labels.type(torch.float),
                      'input_ids1': batch[1],
                      'input_ids2': batch[2],
                      'input_ids3': batch[3],
                      'attention_mask1': batch[4],
                      'attention_mask2': batch[5],
                      'attention_mask3': batch[6],
                      'sentence_count1': batch[7],
                      'sentence_count2': batch[8],
                      'sentence_count3': batch[9],
                      'mode': mode
                      }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        hidden_states = outputs[2]
        loss_val_total += loss.item()

        # hidden_states = hidden_states.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        label_ids = batch[0].cpu().numpy()

        if target == 'post_comment':
            score_ids = batch[7].cpu().numpy()
            index_ids = batch[8].cpu().numpy()
        elif target == 'reply':
            score_ids = batch[4].cpu().numpy()
            index_ids = batch[5].cpu().numpy()
        else:
            score_ids = batch[10].cpu().numpy()
            index_ids = batch[11].cpu().numpy()

        # embeddings.append(hidden_states)
        predictions.append(logits)
        true_vals.append(label_ids)
        true_scores.append(score_ids)
        indexes.append(index_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    # embeddings = np.concatenate(embeddings, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    true_scores = np.concatenate(true_scores, axis=0)
    indexes = np.concatenate(indexes, axis=0)

    accuracy_per_class(predictions, true_vals)

    return loss_val_avg, predictions, true_vals, true_scores, indexes