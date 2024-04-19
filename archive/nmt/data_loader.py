import torch
import json
import numpy as np
import re
import random
import glob
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import source_tokenizer_load
from utils import target_tokenizer_load

import config
DEVICE = config.device


def built_dataset(xml_folder, train_data_path, dev_data_path, max_length, prob=0.85):
    xml_files=glob.glob(f'{xml_folder}/**/*.xml', recursive=True)
    train_data=get_line_pairs(xml_files, max_length, prob)

    random.shuffle(train_data)
    dev_data = train_data[-20:]
    train_data = train_data[:-20]

    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(dev_data_path, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False)


def get_line_pairs(xml_files, max_length, prob=0.85):
    line_pairs=[]
    for file in xml_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = f.read()
            src_lines = re.findall(r'<th>(.*?)</th>', data)
            tgt_lines = re.findall(r'<zh>(.*?)</zh>', data)
            i=0
            while i < len(src_lines):
                if tgt_lines[i]=='':
                    i+=1
                    continue
                src_line = src_lines[i].replace('\\n', '\n')
                tgt_line = tgt_lines[i].replace('\\n', '\n')
                if len(src_line) > max_length:
                    i+=1
                    continue
                while random.random() < prob and i+1<len(src_lines) and tgt_lines[i+1]!='' and len(src_line)+len(src_lines[i+1])+1 <= max_length:
                    i+=1
                    src_line += '\n'+src_lines[i].replace('\\n', '\n')
                    tgt_line += '\n'+tgt_lines[i].replace('\\n', '\n')
                line_pairs.append([src_line, tgt_line])
                i+=1
    return line_pairs


def subsequent_mask(size):
    """Mask out subsequent positions."""
    # Set the shape of the subsequent_mask matrix.
    attn_shape = (1, size, size)

    # Generate a subsequent_mask matrix with ones in the upper-right corner (excluding the main diagonal) and zeros in the lower-left corner (including the main diagonal).
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # Return a subsequent_mask matrix with all elements in the upper-right corner set to False and all elements in the lower-left corner set to True.
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        # Evaluate the non-empty parts of the current input sentence as a boolean sequence.
        # Add an additional dimension in front of the sequence length to create a matrix with dimensions of 1×seq length.
        self.src_mask = (src != pad).unsqueeze(-2)
        # If the output target is not empty, mask the target sentence that the decoder will use.
        if trg is not None:
            trg = trg.to(DEVICE)
            # The target input part used by the decoder.
            self.trg = trg[:, :-1]
            # The predicted target output during decoder training.
            self.trg_y = trg[:, 1:]
            # Apply attention mask to the target input part.
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # Count the actual number of words in the target output that should be produced.
            self.ntokens = (self.trg_y != pad).data.sum()

    # Masking operation
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path):
        self.out_src_sent, self.out_tgt_sent = self.get_dataset(data_path, sort=True)
        self.sp_src = source_tokenizer_load(config.src_vocab_path)[0]
        self.sp_tgt = target_tokenizer_load(config.tgt_vocab_path)[0]
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2

    @staticmethod
    def len_argsort(seq):
        """Pass a series of sentence data (in the form of tokenized lists), sort them by sentence length, and return the sorted indices of the original sentences in the data."""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """Sort both the Chinese and English sentences in the same order, based on the order of (sentence indices) sorted by English sentence length."""
        dataset = json.load(open(data_path, 'r'))
        out_src_sent = []
        out_tgt_sent = []
        for idx in range(len(dataset)):
            src_sent = dataset[idx][0]
            tgt_sent = dataset[idx][1]
            if len(src_sent) < 2000 and len(tgt_sent) < 2000:
                out_src_sent.append(dataset[idx][0])
                out_tgt_sent.append(dataset[idx][1])
        if sort:
            sorted_index = self.len_argsort(out_src_sent)
            out_src_sent = [out_src_sent[i] for i in sorted_index]
            out_tgt_sent = [out_tgt_sent[i] for i in sorted_index]
        return out_src_sent, out_tgt_sent

    def __getitem__(self, idx):
        eng_text = self.out_src_sent[idx]
        chn_text = self.out_tgt_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        return len(self.out_src_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_src(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_tgt(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)
