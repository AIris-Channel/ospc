import os, sys
import torch
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from nmt_model import make_model, batch_greedy_decode
from beam_decoder import beam_search
import sentencepiece as spm

sp_in = spm.SentencePieceProcessor()
sp_in.Load(f'{current_dir}/bpe_models/mix.model')

sp_out = spm.SentencePieceProcessor()
sp_out.Load(f'{current_dir}/bpe_models/en.model')

src_vocab_size = 13000
tgt_vocab_size = 3500
d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = make_model(src_vocab_size, tgt_vocab_size, n_layers,
                    d_model, d_ff, n_heads, dropout)
model.load_state_dict(torch.load(f'{current_dir}/nmt_model.pth'))
model.eval()

max_len = 100
padding_idx = 0
bos_idx = 1
eos_idx = 2
beam_size = 3

def translate(src, model, use_beam=True):
    with torch.no_grad():
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, max_len,
                                           padding_idx, bos_idx, eos_idx,
                                           beam_size, device)
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=max_len)
    translation = [sp_out.DecodeIds(_s).replace('\\n', '\n') for _s in decode_result]
    text = ' ' + translation[0]
    text = text.replace(' i ', ' I ').replace(' i\'', ' I\'')
    return text.strip()

BOS = 1
EOS = 2

def translate_sentence(text):
    text = text.replace('\n', '\\n')
    src_tokens = [[BOS] + sp_in.EncodeAsIds(text) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(device)
    return translate(batch_input, model, use_beam=True)


if __name__ == '__main__':
    import time, json
    data_path = '../benchmark/benchmark_fb_en'
    with open(f'{data_path}/dev.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            start = time.time()
            text = translate_sentence(data['text'])
            end = time.time()
            print(text, 'time:', end - start)
