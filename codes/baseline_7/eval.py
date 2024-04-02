import time
import json
from utils import find_percent, calculate_auroc, calculate_accuracy
from blip_ocr import caption_image, end_caption

data_path = '../benchmark/benchmark_cherrypick_old'
probs = []
labels = []
val_targets = []

captions = []
with open(f'{data_path}/img.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        caption, text = caption_image(f'{data_path}/' + data['img'])
        captions.append([caption, text])
        val_targets.append(data['label'])

end_caption()

from qwen import process_image
for caption, text in captions:
    start = time.time()
    prob = process_image(caption, text)
    probs.append(prob)
    end = time.time()
    print(prob, 'time:', end - start)

threshold = find_percent(probs, 0.5)
for prob in probs:
    labels.append(1 if prob > threshold else 0)

auroc = calculate_auroc(probs, val_targets)
acc = calculate_accuracy(labels, val_targets)
print(f'AUROC: {auroc:.4f}\nAccuracy: {acc:.4f}')
