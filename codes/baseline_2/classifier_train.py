import json, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from classifier import Classifier
from auroc import calculate_auroc

from PIL import Image, ImageOps
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


input_size = 2560 # 50296 2560
hidden_size = 512
num_classes = 2
learning_rate = 1e-5
batch_size = 32
num_epochs = 1000
log_interval = 10
eval_interval = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


user_msg = 'Is it an offensive meme?'
model_path = 'MoE-LLaVA-Phi2-2.7B-4e-384'  # LanguageBind/MoE-LLaVA-Qwen-1.8B-4e or LanguageBind/MoE-LLaVA-StableLM-1.6B-4e
device = 'cuda'
load_4bit, load_8bit = False, False  # FIXME: Deepspeed support 4bit or 8bit?
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
image_processor = processor['image']
conv_mode = "phi"  # qwen or stablelm


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
        return {'hidden_states': hidden_states, 'logits': logits}


class CustomDataset(Dataset):
    def __init__(self, image_paths, targets):
        self.image_paths = image_paths
        self.targets = targets
        self.cache = {}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        if os.path.exists(image_path + '.pth'):
            x = torch.load(image_path + '.pth')
        else:
            x = process_image(image_path)['hidden_states']
            torch.save(x, image_path + '.pth')
        y = self.targets[index]
        y = torch.FloatTensor([y, 1-y])
        return x, y


if __name__ == '__main__':
    image_paths = []
    targets = []
    with open('../datasets/hateful_memes/train.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            image_paths.append('../datasets/hateful_memes/' + data['img'])
            targets.append(data['label'])
    val_image_paths = []
    val_targets = []
    with open('../datasets/hateful_memes/dev.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            val_image_paths.append('../datasets/hateful_memes/' + data['img'])
            val_targets.append(data['label'])
    dataset = CustomDataset(image_paths, targets)
    val_dataset = CustomDataset(val_image_paths, val_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    c_model = Classifier(input_size, hidden_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(c_model.parameters(), lr=learning_rate)

    if os.path.exists('classifier.pth'):
        print('Load classifier.pth')
        c_model.load_state_dict(torch.load('classifier.pth'))
    if os.path.exists('optimizer.pth'):
        print('Load optimizer.pth')
        optimizer.load_state_dict(torch.load('optimizer.pth'))

    total_loss = 0.0
    global_step = 0
    best_score = 0
    for epoch in range(num_epochs):
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device, dtype=torch.float32)
            batch_targets = batch_targets.to(device)

            outputs = c_model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1 

            if global_step % log_interval == 0:
                avg_loss = total_loss / log_interval
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{global_step}], Loss: {avg_loss:.4f}')
                total_loss = 0.0
            
            if global_step % eval_interval == 0:
                answers = []
                with torch.inference_mode():
                    for batch_inputs, batch_targets in val_dataloader:
                        batch_inputs = batch_inputs.to(device, dtype=torch.float32)
                        outputs = c_model(batch_inputs)
                        answers.extend([x[0] for x in outputs.tolist()])
                auroc = calculate_auroc(answers, val_targets)
                print(f'AUROC: {auroc}')
                if auroc > best_score:
                    best_score = auroc
                    print('Save best')
                    torch.save(c_model.state_dict(), 'classifier.pth')
                    torch.save(optimizer.state_dict(), 'optimizer.pth')
