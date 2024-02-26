import sys
original_stdout = sys.stdout
sys.stdout = sys.stderr
from llava_phi import process_image


def find_percent(num_list, percent):
    n = len(num_list)
    index = int(n * percent)
    sorted_a = sorted(num_list)
    if n % 2 == 0:
        return (sorted_a[index - 1] + sorted_a[index]) / 2
    else:
        return sorted_a[index]


if __name__ == '__main__':
    probs = []
    for line in sys.stdin:
        image_path = line.rstrip()
        try:
            prob = process_image(image_path)
            probs.append(prob)
        except Exception as e:
            sys.stderr.write(str(e))
    threshold = find_percent(probs, 0.5)
    for prob in probs:
        label = 1 if prob > threshold else 0
        original_stdout.write(f'{prob:.4f}\t{label}\n')
