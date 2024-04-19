# Neural Machine Translation

Based on [hemingkx / ChineseNMT](https://github.com/hemingkx/ChineseNMT)

## Environment
```bash
pip install -r requirements.txt
```

## Data Processing

### BPE Tokenization

To handle tokenization, the `tools/tokenize.py` script is provided, supporting two operation modes: `train` and `test`.

#### Training a New BPE Model

To train a new BPE model, use the following command:

```bash
python tokenize.py train --input corpus_en.txt --vocab_size 3500 --model_name en --model_type bpe --character_coverage 0.9995
```

- `--input`: A text file containing a large corpus.
- `--vocab_size`: The desired vocabulary size.
- `--model_name`: The name for the generated model.
- `--model_type`: The type of model, which is BPE in this case.
- `--character_coverage`: The percentage of characters covered by the model.

This command generates a model file (`en.model`) in the same directory.

#### Testing the Trained BPE Model

To test the trained BPE model, use the command:

```bash
python tokenize.py test --model_path ./en.model --text "knowing white people , that's probably the baby father"
```

### Training Configuration

To configure the training process, modifications in `config.py` are required:

- `src_vocab_size`: Vocabulary size of the source language.
- `tgt_vocab_size`: Vocabulary size of the target language.
- `src_vocab_path`: Path to the source language's BPE model.
- `tgt_vocab_path`: Path to the target language's BPE model.
- `model_dir`: Output path for the model.
- `train_data_path`: Path to the training data, in JSON format.
- `dev_data_path`: Path to the development data, in JSON format.
- `test_data_path`: Path to the test data, in JSON format.

#### Training Data Format

The training data should be formatted as a list of lists, where each inner list contains a pair of strings: the source sentence and the target sentence. Save this data in a JSON file. Example format:

```json
[
  ["美国海军陆战队第2海军陆战队远征旅，海军陆战队第一营第五局直升机营leatherneck夜晚的空中打击在阿富汗的赫尔曼德省星期四2009年7月2日。", "us marines from the 2nd marine expeditionary brigade, 1st battalion 5th marines board helicopters at camp leatherneck for a night air assault in afghanistan's helmand province thursday july 2, 2009."],
  ["எனவே, இந்த alu 16-பிட் ஆகும், ஆனால் பஸ் இடைமுகத்தில் நம்மிடம் உள்ள இந்த சேர்க்கை இது 20-பிட் ஆகும்.", "so, this alu is 16-bit whereas, this adder that we have in bus interface this is 20-bit."]
]
```

## Training

To start training, run:

```bash
python train.py
```

The script automatically loads the latest checkpoint from `model_dir` specified in `config.py` for continued training.

## Inference

For inference, run:

```bash
python your_script_name.py "中国的传统文化源远流长，包括绘画、音乐、戏剧等多个领域。" --beam_search
```

The script automatically uses the latest checkpoint from `model_dir` for inference.
