import os, sys, json
import paddle
paddle.utils.run_check()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from torch.nn.functional import softmax

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

model_path = f'{current_dir}/../qwen_qlora'
with open(f'{model_path}/adapter_config.json', 'r', encoding='utf-8') as f:
    adapter_config = json.load(f)
adapter_config['base_model_name_or_path'] = f'{current_dir}/../Qwen1.5-14B-Chat-GPTQ-Int4'
with open(f'{model_path}/adapter_config.json', 'w', encoding='utf-8') as f:
    json.dump(adapter_config, f)
device = 'cuda'
torch.set_default_device(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
temperature = 5

def process_image(caption, text):
    system_prompt = ''
    usr_msg = f'Racial Disparities: Memes perpetuating stereotypes or prejudices based on race or ethnicity.\nReligious Beliefs and Practices: Memes that mock or demean specific religions or religious practices.\nSexual Orientation: Memes that promotes negative stereotypes or biases about different sexual orientations.\nNationalistic Sentiments and Xenophobia: Memes that foster negative attitudes towards migrants or glorify extreme nationalistic views.\nSocio Economic Divides: Memes that highlights or ridicules class struggles and socio-economic disparities.\nAge-Related Biases: Memes perpetuating stereotypes or biases based on a person\'s age.\nGender Discrimination: Memes that promotes gender stereotypes or discriminates based on gender.\nDiscrimination Based on Illnesses and Disabilities: Memes that mock or belittle individuals with illnesses or disabilities.\n\nThere is {caption}. The text on the image is "{text}". Is it a meme that contain the above social biases? ("Yes" or "No")'
    prompt = f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{usr_msg}<|im_end|>\n<|im_start|>assistant\n'
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
    from blip_ocr import caption_image
    data_path = '../benchmark/benchmark_cherrypick_old'
    probs = []
    labels = []
    val_targets = []
    with open(f'{data_path}/img.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            start = time.time()
            caption, text = caption_image(f'{data_path}/' + data['img'])
            prob = process_image(caption, text)
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
