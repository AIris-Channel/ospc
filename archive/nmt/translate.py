import argparse
from train import one_sentence_translate

def main():
    parser = argparse.ArgumentParser(description='Translate a sentence from Chinese to another language.')

    parser.add_argument('sentence', type=str, help='The sentence to translate.')
    parser.add_argument('--beam_search', action='store_true', help='Use beam search if this flag is set. Default is False.')

    args = parser.parse_args()

    translation = one_sentence_translate(args.sentence, beam_search=args.beam_search)

    print(translation)

if __name__ == "__main__":
    main()
