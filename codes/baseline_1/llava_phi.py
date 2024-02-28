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
import json
from sklearn.metrics import roc_auc_score, accuracy_score
import shutil
from torch import nn

current_dir = os.path.dirname(os.path.abspath(__file__))

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

disable_torch_init()
user_msg = 'Is it an offensive meme?'
model_path = 'MoE-LLaVA-Phi2-2.7B-4e-384'  # LanguageBind/MoE-LLaVA-Qwen-1.8B-4e or LanguageBind/MoE-LLaVA-StableLM-1.6B-4e
device = 'cuda'
load_4bit, load_8bit = False, False  # FIXME: Deepspeed support 4bit or 8bit?
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
image_processor = processor['image']
conv_mode = "phi"  # qwen or stablelm
temperature = 0.4
THRESOLD = 0.65

classifier = SimpleMLP(input_size=8, hidden_size=16, output_size=1)
classifier_path = os.path.join(current_dir, 'classifier.pth')
classifier.load_state_dict(torch.load(classifier_path))
classifier.to("cuda")
test_prompt_sets = [
    "Is it a meme perpetuating stereotypes or prejudices based on race or ethnicity?",
    "Is it a meme that mock or demean specific religions or religious practices?",
    "Is it a meme that promotes negative stereotypes or biases about different sexual orientations?",
    "Is it a meme that foster negative attitudes towards migrants or glorify extreme nationalistic views?",
    "Is it a meme that highlights or ridicules class struggles and socio-economic disparities?",
    "Is it a meme that perpetuating stereotypes or biases based on a person's age?",
    "Is it a meme that promotes gender stereotypes or discriminates based on gender?",
    "Is it a meme that mock or belittle individuals with illnesses or disabilities?",

]

def process_image(image_path):
    conv = conv_templates[conv_mode].copy()
    image_tensor = image_processor.preprocess(ImageOps.exif_transpose(Image.open(image_path)).convert('RGB'), return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)

    # Initialize variables to store probabilities and the "No" responses count
    yes_probs = []

    for test_prompt in test_prompt_sets:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + test_prompt
        conv = conv_templates[conv_mode].copy()
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
                temperature=temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            
            output = model(**model_inputs, return_dict=True)
            logits = output.logits[:, -1, :]  # 获取最后一个token的logits

        # Convert tokens to ids and extract logits for affirmative and negative tokens
        affirmative_ids = tokenizer.convert_tokens_to_ids(['ĠYes'])
        negative_ids = tokenizer.convert_tokens_to_ids(['ĠNo'])
        affirmative_logits = logits[:, affirmative_ids] / temperature
        negative_logits = logits[:, negative_ids] / temperature

        # Compute probabilities
        probs = softmax(torch.stack((torch.mean(affirmative_logits, dim=1), torch.mean(negative_logits, dim=1))), dim=0)

        yes_probs.append(probs[0].item())


    yes_probs_tensor = torch.tensor(yes_probs).float().unsqueeze(0)  # Reshape for single sample
    with torch.no_grad():
        yes_probs_tensor = yes_probs_tensor.to('cuda')
        predicted_probs = classifier(yes_probs_tensor)
        final_prob = torch.sigmoid(predicted_probs).item()  # Get final probability

    return final_prob

def test_model(test_image_sets, test_image_labels):
    predictions, labels = [], []
    
    # Directory for images to be evaluated
    to_evaluate_dir = os.path.join(test_image_sets, 'to_evaluate')
    
    # Clear the to_evaluate directory if it exists
    if os.path.exists(to_evaluate_dir):
        shutil.rmtree(to_evaluate_dir)
    os.makedirs(to_evaluate_dir)
    
    with open(test_image_labels, 'r') as file:
        for line in file:
            data = json.loads(line)
            image_path = os.path.join(test_image_sets, data['img'][4:])
            true_label = data['label']
            
            # Process image and compute prediction result
            process_result = process_image(image_path)
            
            # Collect predictions and true labels
            predictions.append(process_result)
            labels.append(true_label)
            
            # If the prediction is incorrect, copy the image to the to_evaluate directory
            predicted_label = 1 if process_result > 0.5 else 0
            if predicted_label != true_label:
                # Determine the filename
                img_filename = os.path.basename(image_path)
                # Set the path for the new location to copy
                eval_image_path = os.path.join(to_evaluate_dir, img_filename)
                # Copy the image to the to_evaluate directory
                shutil.copy(image_path, eval_image_path)
    

    # Calculate AUROC and accuracy
    auroc = roc_auc_score(labels, predictions)
    accuracy = accuracy_score(labels, [1 if p > 0.5 else 0 for p in predictions])
    
    return auroc, accuracy

if __name__ == '__main__':
    # test_image_sets = '../local_test/test_images/'
    
    test_image_sets = '../benchmark/benchmark_fb_en/img/'
    test_image_labels = '../benchmark/benchmark_fb_en/dev.jsonl'
    # 确保路径正确
    if not os.path.exists(test_image_sets):
        print(f"Directory {test_image_sets} does not exist.")
    else:
        start_time = time.time()
        auroc, accuracy = test_model(test_image_sets, test_image_labels)
        end_time = time.time()
        print(f"Model AUROC: {auroc}, Accuracy: {accuracy}")
        print(f"Time consumed:{end_time - start_time}s")
