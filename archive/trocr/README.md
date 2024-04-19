# TrOCR

Based on [chineseocr / trocr-chinese](https://github.com/chineseocr/trocr-chinese)

This project aims to build an OCR dataset and train a TrOCR model capable of recognizing English, Chinese, Malay, and Tamil by printing text from a corpus on random images with random fonts, sizes, and colors.

## Environment Setup

```bash
pip install -r requirements.txt
```

## Download Pre-trained TrOCR Model Weights

You can clone the pre-trained TrOCR model weights from [CjangCjengh/ezmt_pretrained](https://huggingface.co/CjangCjengh/ezmt_pretrained)
```bash
git clone https://huggingface.co/CjangCjengh/ezmt_pretrained weights
```

## Modify Pre-trained Model's Embedding Layer Vocabulary

To modify the vocabulary size of the pre-trained model's embedding layer, use the following command:

```bash
python init_model.py \
    --vocab_path ./vocab/vocab.model \
    --pretrain_model ./weights \
    --weights_path ./new_weights
```

- `vocab_path`: The model file obtained from BPE tokenization.
- `pretrain_model`: The folder path of the pre-trained model weights.
- `weights_path`: The folder path where the modified model weights will be saved.

## Training

To train the model, use the following command:

```bash
python train.py --weights_path="./new_weights" --checkpoint_path="./checkpoints" --dataset_path="./image/*/*.jpg" --text_path="./text/train.json" --per_device_train_batch_size=32 --per_device_eval_batch_size=32 --num_train_epochs=10 --save_steps=100
```

- `weights_path`: The folder path of the initial weights.
- `checkpoint_path`: The folder path where checkpoints will be saved.
- `dataset_path`: The path to the images used for printing text.
- `text_path`: The corpus used for constructing the OCR data, similar to the data used for training Neural Machine Translation.

## Testing

To test the model, use the following command:

```bash
python test.py --weight_dir="./new_weights" --image_path="./test.jpg"
```

- `weight_dir`: The folder path of the model weights.
