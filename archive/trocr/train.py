import os
import json
import argparse
import random
from glob import glob
from dataset import trocrDataset, decode_text
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import default_data_collator
from sklearn.model_selection import train_test_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import sentencepiece as spm


def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                insert = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                replace = dp[i - 1][j - 1] + 1
                dp[i][j] = min(insert, delete, replace)

    return dp[m][n]

def compute_metrics(pred):
    '''
    calculate cer, acc
    :param pred:
    :return:
    '''
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = [decode_text(pred_id, vocab, vocab_inp, sp) for pred_id in pred_ids]
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = [decode_text(labels_id, vocab, vocab_inp, sp) for labels_id in labels_ids]

    cer = [edit_distance(pred, label) for pred, label in zip(pred_str, label_str)]
    cer = sum(cer) / len(cer)
    acc = [pred == label for pred, label in zip(pred_str, label_str)]
    acc = sum(acc) / len(acc)

    return {'cer': cer, 'acc': acc}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', default='weights', type=str)
    parser.add_argument('--checkpoint_path', default='new_weights', type=str)
    parser.add_argument('--dataset_path', default='data/*/*.jpg', type=str)
    parser.add_argument('--text_path', default='data/*/*.json', type=str)
    parser.add_argument('--per_device_train_batch_size', default=32, type=int)
    parser.add_argument('--per_device_eval_batch_size', default=8, type=int)
    parser.add_argument('--max_target_length', default=128, type=int)

    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--eval_steps', default=200, type=int)
    parser.add_argument('--save_steps', default=100, type=int)

    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1', type=str)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    print('loading data')
    paths = glob(args.dataset_path)

    train_paths, test_paths = train_test_split(paths, test_size=0.001)
    print('train num:', len(train_paths), 'test num:', len(test_paths))
    
    text_paths = glob(args.text_path)
    text_pairs = []
    for text_path in text_paths:
        with open(text_path, 'r', encoding='utf-8') as f:
            new_pairs = json.load(f)
        for x, y in new_pairs:
            if len(x) < 400:
                text_pairs.append(x)
    
    test_text = random.sample(text_pairs, k=100)
    print('Text loaded')

    # Image preprocessing
    processor = TrOCRProcessor.from_pretrained(args.weights_path)
    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}
    transformer = lambda x: x

    sp = spm.SentencePieceProcessor()
    sp.Load(f'{args.weights_path}/sentencepiece.bpe.model')
    train_dataset = trocrDataset(paths=train_paths, processor=processor, max_target_length=args.max_target_length, transformer=transformer, text_pairs=text_pairs, sp=sp)
    eval_dataset = trocrDataset(paths=test_paths, processor=processor, max_target_length=args.max_target_length, transformer=transformer, text_pairs=test_text, sp=sp)

    model = VisionEncoderDecoderModel.from_pretrained(args.weights_path)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 256
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        fp16=True,
        output_dir=args.checkpoint_path,
        logging_steps=100,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        report_to=[]
    )

    # seq2seq trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_path, 'last'))
    processor.save_pretrained(os.path.join(args.checkpoint_path, 'last'))
