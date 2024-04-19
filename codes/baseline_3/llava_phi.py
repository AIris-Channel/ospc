import os, sys, json
import torch
from PIL import Image, ImageOps
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from torch.nn.functional import softmax

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trocr import recognize_image
from nmt import translate_sentence

disable_torch_init()
model_path = 'MoE-LLaVA-Phi2-2.7B-4e-384'
device = 'cuda'
load_4bit, load_8bit = False, False  # FIXME: Deepspeed support 4bit or 8bit?
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
image_processor = processor['image']
conv_mode = "phi"  # qwen or stablelm
temperature = 1

def process_image(image_path):
    conv = conv_templates[conv_mode].copy()
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    text = recognize_image(image)
    text = translate_sentence(text)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)

    user_msg = f'The text on the image means "{text}". Is it an offensive meme?'
    inp = DEFAULT_IMAGE_TOKEN + '\n' + user_msg
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        model_inputs = model.prepare_inputs_for_generation(input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
        output = model(**model_inputs, output_hidden_states=True, return_dict=True)
        logits = output.logits[:, -1, :]  # 获取最后一个token的logits

        # Convert tokens to ids and extract logits for affirmative and negative tokens
        affirmative_ids = tokenizer.convert_tokens_to_ids(['ĠYes'])
        negative_ids = tokenizer.convert_tokens_to_ids(['ĠNo'])
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
