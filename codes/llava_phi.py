import re
import torch
from PIL import Image, ImageOps
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from torch.nn.functional import softmax
import os,time
from classifier import Classifier

disable_torch_init()
user_msg = 'Is it an offensive meme?'
model_path = 'MoE-LLaVA-Phi2-2.7B-4e-384'  # LanguageBind/MoE-LLaVA-Qwen-1.8B-4e or LanguageBind/MoE-LLaVA-StableLM-1.6B-4e
device = 'cuda'
load_4bit, load_8bit = False, False  # FIXME: Deepspeed support 4bit or 8bit?
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
image_processor = processor['image']
conv_mode = "phi"  # qwen or stablelm
temperature = 0.2

c_model = Classifier(2560, 512, 2).to(device)
c_model.load_state_dict(torch.load('classifier.pth'))

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
    # test_image_sets = '../local_test/test_images/'
    
    test_image_sets = '../datasets/hateful_memes/img'
    # 确保路径正确
    if not os.path.exists(test_image_sets):
        print(f"Directory {test_image_sets} does not exist.")
    else:
        # __import__('ipdb').set_trace()
        # 列出目录中的所有文件
        image_files = [f for f in os.listdir(test_image_sets) if os.path.isfile(os.path.join(test_image_sets, f))]
        
        for image_file in image_files:
            image_path = os.path.join(test_image_sets, image_file)
            
            # 开始计时
            start_time = time.time()
            
            # 处理图片并计算需要的信息
            print(f"Processing {image_file}...")
            process_result = process_image(image_path)
            
            # 结束计时
            end_time = time.time()
            
            # 打印处理结果和处理时间
            print(f"Processed {image_file} in {end_time - start_time} seconds. Result: {process_result}")

