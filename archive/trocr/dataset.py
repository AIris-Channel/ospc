import re
import random
from PIL import Image
from torch.utils.data import Dataset
import torch
from print_data import process_image_en, process_image_ta, process_image_zh


class trocrDataset(Dataset):
    def __init__(self, paths, processor, max_target_length=128, transformer=lambda x:x, text_pairs=None, sp=None):
        self.paths = paths
        self.processor = processor
        self.sp = sp
        self.transformer = transformer
        self.max_target_length = max_target_length
        self.nsamples = len(text_pairs)
        self.vocab = processor.tokenizer.get_vocab()
        self.text_pairs = text_pairs

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
            image_file = self.paths[idx % len(self.paths)]
            image = Image.open(image_file).convert("RGB")
            text = self.text_pairs[idx]
            labels = encode_text(text, max_target_length=self.max_target_length, vocab=self.vocab, sp=self.sp)
            x = random.random()
            if x < 0.3:
                text = text.upper()
            elif x < 0.4:
                text = text.title()
            elif x < 0.7:
                text = text.capitalize()
            if re.search(r'[\u4e00-\u9fff]', text):
                image = process_image_zh(image, text)
            elif re.search(r'[\u0b80-\u0bff]', text):
                image = process_image_ta(image, text)
            else:
                image = process_image_en(image, text)
            image = self.transformer(image) # Image enhancement functions
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

            return encoding


def encode_text(text, max_target_length=128, vocab=None, sp=None):
    """
    Custom list: ['<td>',"3","3",'</td>',....]
    {'input_ids': [0, 1092, 2, 1, 1],
    'attention_mask': [1, 1, 1, 0, 0]}
    """
    tokens = [vocab.get('<s>')]
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    mask = []

    pieces = sp.EncodeAsPieces(text)
    for p in pieces[:max_target_length - 2]:
        tokens.append(vocab[p])
        mask.append(1)

    tokens.append(vocab.get('</s>'))
    mask.append(1)

    if len(tokens) < max_target_length:
        for i in range(max_target_length - len(tokens)):
            tokens.append(pad)
            mask.append(0)

    return tokens
    #return {"input_ids": tokens, 'attention_mask': mask}


def decode_text(tokens, vocab, vocab_inp, sp):
    # decode trocr
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    text = ''
    tokens = [tk.item() - 1 for tk in tokens if tk not in [s_end, s_start , pad, unk]]
    text = sp.DecodeIds(tokens)
    return text
