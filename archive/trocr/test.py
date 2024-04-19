import argparse
import torch
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sentencepiece as spm


def decode_text(tokens, vocab, vocab_inp, sp):
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    text = ''
    tokens = [tk.item() - 1 for tk in tokens if tk not in [s_end, s_start , pad, unk]]
    text = sp.DecodeIds(tokens)
    return text

def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                insert = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                replace = dp[i - 1][j - 1] + 1
                dp[i][j] = min(insert, delete, replace)

    return dp[m][n]


def main(weight_dir, image_path):
    # Set the device to cuda if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model and processor
    model = VisionEncoderDecoderModel.from_pretrained(weight_dir).to(device)
    model.eval()
    processor = TrOCRProcessor.from_pretrained(weight_dir)

    # Load vocabulary and prepare for decoding
    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}
    sp = spm.SentencePieceProcessor()
    sp.Load(f'{weight_dir}/sentencepiece.bpe.model')

    # Process the image
    img = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    pixel_values = processor([img], return_tensors='pt').pixel_values

    # Generate text from image
    with torch.no_grad():
        generated_ids = model.generate(pixel_values.to(device))
    text = decode_text(generated_ids[0].cpu().numpy(), vocab, vocab_inp, sp)

    print(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images to extract text using TrOCR.")

    parser.add_argument("--weight_dir", type=str, required=True, help="Directory where the model weights are stored.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file.")

    args = parser.parse_args()

    main(args.weight_dir, args.image_path)
