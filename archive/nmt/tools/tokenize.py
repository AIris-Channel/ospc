import argparse
import sentencepiece as spm


def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """
    search on https://github.com/google/sentencepiece/blob/master/doc/options.md to learn more about the parameters
    :param input_file: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
                       By default, SentencePiece normalizes the input with Unicode NFKC.
                       You can pass a comma-separated list of files.
    :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
    :param model_name: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
    :param model_type: model type. Choose from unigram (default), bpe, char, or word.
                       The input sentence must be pretokenized when using word type.
    :param character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with
                               rich character set like Japanse or Chinese and 1.0 for other languages with
                               small character set.
    """
    input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s --allow_whitespace_only_pieces=true --byte_fallback ' \
                     '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage)
    spm.SentencePieceTrainer.Train(cmd)


def test(model_path, text):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))


def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece model or test an existing model.")
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    parser_train = subparsers.add_parser('train', help='Train a new model')
    parser_train.add_argument('--input', type=str, required=True, help='Input file path')
    parser_train.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size')
    parser_train.add_argument('--model_name', type=str, required=True, help='Model name')
    parser_train.add_argument('--model_type', type=str, required=True, choices=['bpe', 'unigram', 'char', 'word'], help='Model type')
    parser_train.add_argument('--character_coverage', type=float, required=True, help='Character coverage')

    parser_test = subparsers.add_parser('test', help='Test an existing model')
    parser_test.add_argument('--model_path', type=str, required=True, help='Path to the SentencePiece model file')
    parser_test.add_argument('--text', type=str, required=True, help='Text to encode')

    args = parser.parse_args()

    if args.command == 'train':
        train(args.input, args.vocab_size, args.model_name, args.model_type, args.character_coverage)
    elif args.command == 'test':
        test(args.model_path, args.text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
