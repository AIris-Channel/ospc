import os, re, sys, json
import torch
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from torch.nn.functional import softmax

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trocr import recognize_image
# from nmt import translate_sentence
from blip import blip_caption

model_path = f'{current_dir}/../Qwen1.5-14B-Chat-GPTQ-Int4'
device = 'cuda'
torch.set_default_device(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
temperature = 5

def process_image(image_path):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    caption = blip_caption(image)
    text = recognize_image(image)
    # if re.search(r'[\u0b80-\u0bff]', text):
    #     text = translate_sentence(text)
    system_prompt = 'You are a meme censor, and you are skilled at determining whether a meme is offensive.'
    # usr_msg = f'There is {caption}. The text on the image is "{text}". Is it an offensive meme?'
    usr_msg2 = f'Racial Disparities: Memes perpetuating stereotypes or prejudices based on race or ethnicity.\nReligious Beliefs and Practices: Memes that mock or demean specific religions or religious practices.\nSexual Orientation: Memes that promotes negative stereotypes or biases about different sexual orientations.\nNationalistic Sentiments and Xenophobia: Memes that foster negative attitudes towards migrants or glorify extreme nationalistic views.\nSocio Economic Divides: Memes that highlights or ridicules class struggles and socio-economic disparities.\nAge-Related Biases: Memes perpetuating stereotypes or biases based on a person\'s age.\nGender Discrimination: Memes that promotes gender stereotypes or discriminates based on gender.\nDiscrimination Based on Illnesses and Disabilities: Memes that mock or belittle individuals with illnesses or disabilities.\n\nThere is {caption}. The text on the image is "{text}". Is it a meme that contain the above social biases or otherwise portraying in a negative light the above categories?'
    prompt = f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{usr_msg2}<|im_end|>\n<|im_start|>assistant\n'
    inputs = tokenizer(prompt, return_tensors='pt', return_attention_mask=False)

    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, return_dict=True)
        logits = output.logits[:, -1, :]  # 获取最后一个token的logits

        # Convert tokens to ids and extract logits for affirmative and negative tokens
        affirmative_ids = tokenizer.encode('Yes')
        negative_ids = tokenizer.encode('No')
        affirmative_logits = logits[:, affirmative_ids] / temperature
        negative_logits = logits[:, negative_ids] / temperature

        # Compute probabilities
        probs = softmax(torch.stack((torch.mean(affirmative_logits, dim=1), torch.mean(negative_logits, dim=1))), dim=0)

    return probs[0].item()


if __name__ == '__main__':
    import time
    from utils import find_percent, calculate_auroc, calculate_accuracy
    data_path = '../benchmark/benchmark_cherrypick'
    probs = []
    labels = []
    val_targets = []
    with open(f'{data_path}/img.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            start = time.time()
            prob = process_image(f'{data_path}/' + data['img'])
            probs.append(prob)
            end = time.time()
            print(prob, 'time:', end - start)
            val_targets.append(data['label'])

    threshold = find_percent(probs, 0.5)
    for prob in probs:
        labels.append(1 if prob > threshold else 0)

    auroc = calculate_auroc(probs, val_targets)
    acc = calculate_accuracy(labels, val_targets)
    print(f'AUROC: {auroc:.4f}\nAccuracy: {acc:.4f}')
