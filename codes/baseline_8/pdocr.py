import os, sys, json
from paddleocr import PaddleOCR

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

ocr = PaddleOCR(det_model_dir=f'{current_dir}/ch_PP-OCRv4_det_server_infer',
                rec_model_dir=f'{current_dir}/ch_PP-OCRv4_rec_server_infer')


def recognize_image(image_path):
    result = ocr.ocr(image_path)
    result = result[0]
    if result:
        text = ' '.join([line[-1][0] for line in result])
        prob = [line[-1][-1] for line in result]
        prob = sum(prob) / len(prob)
        return text, prob
    else:
        return '', 0

def end_pdocr():
    global ocr
    del ocr


if __name__ == '__main__':
    import time
    data_path = '../benchmark/benchmark_fb_en'
    with open(f'{data_path}/dev.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            start = time.time()
            text, prob = recognize_image(f'{data_path}/' + data['img'])
            end = time.time()
            print(text, prob, 'time:', end - start)
