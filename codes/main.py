import sys
original_stdout = sys.stdout
sys.stdout = sys.stderr


def find_percent(num_list, percent):
    n = len(num_list)
    index = int(n * percent)
    sorted_a = sorted(num_list)
    if n % 2 == 0:
        return (sorted_a[index - 1] + sorted_a[index]) / 2
    else:
        return sorted_a[index]


if __name__ == '__main__':
    from baseline_7.blip_ocr import caption_image, end_caption
    probs = []
    tasks = []
    for line in sys.stdin:
        image_path = line.rstrip()
        try:
            caption, text = caption_image(image_path)
            tasks.append([caption, text])
        except Exception as e:
            sys.stderr.write(str(e))
    
    end_caption()

    from baseline_7.qwen import process_image
    for caption, text in tasks:
        try:
            prob = process_image(caption, text)
            probs.append(prob)
        except Exception as e:
            sys.stderr.write(str(e))

    threshold = find_percent(probs, 0.5)

    for prob in probs:
        label = 1 if prob > threshold else 0
        original_stdout.write(f'{prob:.4f}\t{label}\n')
