from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, accuracy_score
import json
import sys
import numpy as np

def read_jsonl(path):
    data = []
    with open(path) as fin:
        for line in fin:
            label = 0 if json.loads(line)['label'] == 'CORRECT' else 1
            data.append(label)
    return data

def read_txt(path):
    data = []
    with open(path) as fin:
        for line in fin:
            label = 0 if line.strip().split()[0] == 'CORRECT' else 1
            data.append(label)
    return data


def p_r_f1(labels, preds):
    tp, fp, fn, tn = 0, 0, 0, 0
    for label, pred in zip(labels, preds):
        if label == 1 and pred == 1: tp += 1
        elif label == 1 and pred == 0: fn += 1
        elif label == 0 and pred == 1: fp += 1
        else: tn += 1
    print(tp, fp, fn, tn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


if __name__ == '__main__':
    filenames = sys.argv[1:]
    print(filenames)
    all_labels = []
    all_preds = []
    for i in range(0, len(filenames), 2):
        labels = read_jsonl(filenames[i])
        preds = read_txt(filenames[i + 1])
        assert(len(preds) == len(labels))
        all_labels += labels
        all_preds += preds
    acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    print(acc, bacc, p, r, f1)