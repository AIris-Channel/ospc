import os
import re
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = f'{current_dir}/../blip-image-captioning-large'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(device)


def blip_caption(image):
    # Conditional image captioning
    inputs = processor(image, text='this is an image', return_tensors="pt").to(device, torch.float16)

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    caption = re.sub(r'^this is\s*','',caption)
    return caption

def end_blip():
    global model
    del model


if __name__ == '__main__':
    import time, json
    from PIL import Image
    data_path = '../benchmark/benchmark_fb_en'
    with open(f'{data_path}/dev.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            start = time.time()
            image = Image.open(f'{data_path}/' + data['img']).convert('RGB')
            text = blip_caption(image)
            end = time.time()
            print(text, 'time:', end - start)
