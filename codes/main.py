import sys
import logging
from llava_phi import process_image


logging.getLogger().setLevel(logging.CRITICAL)


if __name__ == "__main__":
    for line in sys.stdin:
        image_path = line.rstrip()
        try:
            proba = process_image(image_path)
            if proba > 0.5:
                label = 1
            else:
                label = 0
            sys.stdout.write(f"{proba:.4f}\t{label}\n")
        except Exception as e:
            sys.stderr.write(str(e))
