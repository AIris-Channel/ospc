import os
import random
from PIL import Image, ImageDraw, ImageFont


def random_bool(threshold=0.5):
    return random.random() < threshold

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def find_fonts(folder_path):
    fonts_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.otf', '.ttf')):
                fonts_list.append(os.path.join(root, file))
    return fonts_list

def random_font(folder_path):
    current_dir = os.path.dirname(__file__)
    font_path = random.choice(find_fonts(folder_path))
    font_path = f'{current_dir}/{font_path}'
    return font_path


def process_image(canvas, text, font_dir, line_max_chars, split_by_space=True):
    # Random scaling
    width = int(384 * random.uniform(0.8, 1.2))
    height = int(384 * random.uniform(0.8, 1.2))
    if random_bool(0.05):
        canvas = Image.new('RGB', (width, height), random_color())
    else:
        canvas = canvas.resize((width, height))

    # Random line breaks
    n_chars = len(text)
    min_lines = (n_chars + line_max_chars - 1) // line_max_chars
    if min_lines < 10:
        n_lines = random.randint(min_lines, 10)
    else:
        n_lines = min_lines
    line_min_chars = (n_chars + n_lines - 1) // n_lines

    lines = []
    if split_by_space:
        words = text.split()
    else:
        words = list(text)
    
    if len(words) == 0:
        return canvas

    line_n_chars = random.randint(line_min_chars, line_max_chars)
    current_line = words[0]
    for word in words[1:]:
        if len(current_line + word) < line_n_chars:
            if split_by_space:
                current_line += ' ' + word
            else:
                current_line += word
        else:
            lines.append(current_line)
            while random_bool(0.1):
                lines.append('\n')
            current_line = word
            line_n_chars = random.randint(line_min_chars, line_max_chars)
    lines.append(current_line)
    
    x = random.random()
    max_line_len = max([len(l) for l in lines])
    if x < 0.5:
        lines = [' ' * ((max_line_len - len(l)) // 2) + l for l in lines]
    elif x < 0.6:
        lines = [' ' * (max_line_len - len(l)) + l for l in lines]

    text = '\n'.join(lines)

    # Random font
    font_color = random_color()
    font_path = random_font(font_dir)
    font_size = random.randint(20, 200)
    font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.RAQM)
    text_width, text_height = font.getsize_multiline(text)
    scale = max(text_width / width, text_height / height)
    while scale > 1:
        max_font_size = int(font_size / scale)
        font_size = max_font_size
        font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.RAQM)
        text_width, text_height = font.getsize_multiline(text)
        scale = max(text_width / width, text_height / height)

    # Random position
    draw = ImageDraw.Draw(canvas)
    max_x = max(0, width - text_width)
    max_y = max(0, height - text_height)
    text_position = (random.randint(0, max_x), random.randint(0, max_y))
    if random_bool():
        draw.text(text_position, text, font=font, fill=font_color)
    else:
        border_width = random.randint(1, (font_size + 9) // 10)
        draw.text(text_position, text, font=font, fill=font_color, stroke_width=border_width, stroke_fill=random_color())
    
    # Random scaling
    width = int(width * random.uniform(0.8, 1.2))
    height = int(height * random.uniform(0.8, 1.2))
    canvas = canvas.resize((width, height))
    return canvas


def process_image_en(canvas, text):
    return process_image(canvas, text, 'fonts/english', 30)

def process_image_zh(canvas, text):
    return process_image(canvas, text, 'fonts/chinese', 20, False)

def process_image_ta(canvas, text):
    return process_image(canvas, text, 'fonts/tamil', 20)


if __name__ == '__main__':
    canvas = Image.open('test.jpg')
    text = 'he had no doubt about which vision would win out'
    canvas = process_image_en(canvas, text)
    canvas.save('test_.jpg')
