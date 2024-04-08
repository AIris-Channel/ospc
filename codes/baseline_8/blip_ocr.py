import os, re, sys, json
import paddle
paddle.utils.run_check()
import torch
from PIL import Image, ImageOps

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from pdocr import recognize_image as recognize_image_pd, end_pdocr
from trocr import recognize_image, end_trocr
from nmt import translate_sentence, end_nmt
from blip import blip_caption, end_blip


def caption_image(image_path):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    caption = blip_caption(image)
    text, text_prob = recognize_image_pd(image_path)
    if text_prob < 0.9:
        text = recognize_image(image)
    if re.search(r'[\u0b80-\u0bff]', text):
        text = translate_sentence(text)
    return caption, text

def end_caption():
    end_pdocr()
    end_trocr()
    end_nmt()
    end_blip()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import time
    data_path = '../benchmark/benchmark_cherrypick_old'
    probs = []
    labels = []
    val_targets = []
    with open(f'{data_path}/img.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            start = time.time()
            caption, text = caption_image(f'{data_path}/' + data['img'])
            end = time.time()
            print(caption)
            print(text)
            print('time:', end - start)
            val_targets.append(data['label'])
