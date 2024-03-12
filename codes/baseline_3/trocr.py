import os, sys, json
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = f'{current_dir}/trocr_weights'
processor = TrOCRProcessor.from_pretrained(model_path)
vocab = processor.tokenizer.get_vocab()

vocab_inp = {vocab[key]: key for key in vocab}
model = VisionEncoderDecoderModel.from_pretrained(model_path)
model.eval()
model.to(device)

vocab = processor.tokenizer.get_vocab()
vocab_inp = {vocab[key]: key for key in vocab}


def decode_text(tokens, vocab, vocab_inp):
    ##decode trocr
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    text = ''
    for tk in tokens:
        if tk not in [s_end, s_start , pad, unk]:
           text += vocab_inp[tk]
    text = text.replace('▁', ' ').strip()
    return text


def recognize_image(image):
    pixel_values = processor([image], return_tensors='pt').pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values.to(device))

    generated_text = decode_text(generated_ids[0].cpu().numpy(), vocab, vocab_inp)
    return generated_text


if __name__ == '__main__':
    import time
    data_path = '../benchmark/benchmark_fb_en'
    with open(f'{data_path}/dev.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            start = time.time()
            image = Image.open(f'{data_path}/' + data['img']).convert('RGB')
            text = recognize_image(image)
            end = time.time()
            print(text, 'time:', end - start)
