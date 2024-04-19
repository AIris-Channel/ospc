import os
import json
import shutil
import argparse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoConfig
import sentencepiece as spm


def read_vocab(vocab_path):
    other = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    vocab = {}
    for ot in other:
        vocab[ot] = len(vocab)

    sp = spm.SentencePieceProcessor()
    sp.Load(vocab_path)

    vocab_size = sp.GetPieceSize()
    for i in range(4, vocab_size):
        vocab[sp.IdToPiece(i)] = len(vocab)

    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='vocab/vocab.model', type=str)
    parser.add_argument('--pretrain_model', default='weights', type=str)
    parser.add_argument('--weights_path', default='new_weights', type=str)
    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained(args.pretrain_model)
    pre_model = VisionEncoderDecoderModel.from_pretrained(args.pretrain_model)

    pre_vocab = processor.tokenizer.get_vocab()
    vocab_dict = read_vocab(args.vocab_path)

    keep_tokens = []
    unk_index = pre_vocab.get('<unk>')
    for key in vocab_dict:
        keep_tokens.append(pre_vocab.get(key, unk_index))

    processor.save_pretrained(args.weights_path)
    pre_model.save_pretrained(args.weights_path)

    # Replace word library
    with open(os.path.join(args.weights_path, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False)

    # Replace model parameters
    with open(os.path.join(args.weights_path, 'config.json')) as f:
        model_config = json.load(f)

    # Replace Roberta embedding layer word library
    model_config['decoder']['vocab_size'] = len(vocab_dict)

    # Replace attention vocabulary
    model_config['vocab_size'] = len(vocab_dict)

    with open(os.path.join(args.weights_path, 'config.json'), 'w') as f:
        json.dump(model_config, f, ensure_ascii=False)

    # Load model
    cust_config = AutoConfig.from_pretrained(args.weights_path)
    cust_model = VisionEncoderDecoderModel(cust_config)

    pre_model_weigths = pre_model.state_dict()
    cust_model_weigths = cust_model.state_dict()

    # Init weight
    print('loading init weights')
    for key in pre_model_weigths:
        print('name:', key)
        if pre_model_weigths[key].shape != cust_model_weigths[key].shape:
            wt = pre_model_weigths[key][keep_tokens, :]
            cust_model_weigths[key] = wt
        else:
            cust_model_weigths[key] = pre_model_weigths[key]

    cust_model.load_state_dict(cust_model_weigths)
    cust_model.save_pretrained(args.weights_path)
    shutil.copy(args.vocab_path, f'{args.weights_path}/sentencepiece.bpe.model')
    os.remove(f'{args.weights_path}/tokenizer.json')
