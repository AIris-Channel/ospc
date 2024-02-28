import os, sys, json
import torch
from PIL import Image, ImageOps
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from classifier import Classifier
c_model_path = os.path.join(current_dir, 'classifier.pth')

disable_torch_init()
user_msg = 'Is it an offensive meme?'
model_path = 'MoE-LLaVA-Phi2-2.7B-4e-384'
device = 'cuda'
load_4bit, load_8bit = False, False  # FIXME: Deepspeed support 4bit or 8bit?
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
image_processor = processor['image']
conv_mode = "phi"  # qwen or stablelm
temperature = 0.2

c_model = Classifier(2560, 512, 2).to(device)
c_model.load_state_dict(torch.load(c_model_path))

def process_image(image_path):
    conv = conv_templates[conv_mode].copy()
    image_tensor = image_processor.preprocess(ImageOps.exif_transpose(Image.open(image_path)).convert('RGB'), return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)

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
        hidden_states = output.hidden_states[-1][0, -1]
        logits = output.logits[0, -1]
        probs = c_model(hidden_states.unsqueeze(0).to(device, dtype=torch.float32))
        probs = probs.squeeze()

    return probs[0].item()


if __name__ == '__main__':
    from utils import find_percent, calculate_auroc, calculate_accuracy
    input_size = 2560
    hidden_size = 512
    num_classes = 2
    batch_size = 32
    data_path = '../../benchmark/benchmark_fb_en'
    probs = []
    labels = []
    val_targets = []
    with open(f'{data_path}/dev.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            probs.append(process_image(f'{data_path}/' + data['img']))
            val_targets.append(data['label'])

    threshold = find_percent(probs, 0.5)
    for prob in probs:
        labels.append(1 if prob > threshold else 0)

    auroc = calculate_auroc(probs, val_targets)
    acc = calculate_accuracy(labels, val_targets)
    print(f'AUROC: {auroc:.4f}\nAccuracy: {acc:.4f}')
 