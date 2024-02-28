import numpy as np


def find_percent(num_list, percent):
    n = len(num_list)
    index = int(n * percent)
    sorted_a = sorted(num_list)
    if n % 2 == 0:
        return (sorted_a[index - 1] + sorted_a[index]) / 2
    else:
        return sorted_a[index]


def calculate_auroc(predictions, ground_truth):
    # Sorting the predictions by their scores
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_labels = np.array(ground_truth)[sorted_indices]

    # Calculate the number of positive samples
    num_positives = np.sum(sorted_labels == 1)
    num_negatives = len(sorted_labels) - num_positives

    # Initialize variables for AUROC calculation
    tp_count = 0  # True positives
    fp_count = 0  # False positives
    auroc = 0.0

    # Calculate AUROC
    for label in sorted_labels:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
            auroc += tp_count / num_positives * (1 / num_negatives)
    
    return auroc

def calculate_accuracy(answers, gts):
    correct = sum(1 for pred, gt in zip(answers, gts) if pred == gt)
    total = len(gts)
    accuracy = correct / total if total != 0 else 0
    return accuracy
