import os
import imageio
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Image as IPImage
from io import BytesIO
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor

def visualize_video_sample(data_dir, video_id):
    """ visualizes selected video sample as gif """
    rgb_files = sorted([f for f in os.listdir(data_dir)
                   if f.startswith(f"rgb_{video_id:05d}")])
    
    images = []
    for filename in rgb_files:
        img = Image.open(os.path.join(data_dir, filename))
        images.append(img)

    with BytesIO() as buffer:
        imageio.mimsave(buffer, images, format='GIF', duration=0.2)
        buffer.seek(0)
        display(IPImage(data=buffer.getvalue()))

def _to_pil_list(seq: torch.Tensor):
    # seq: [T, C, H, W] -> list[PIL]
    seq = seq.detach().cpu().clamp(0, 1)
    return [to_pil_image(x) for x in seq]

def draw_rollout_grid(ctx_seq, fut_seq, pred_seq, writer, dir_imgs, iter_, mode='train'):
    """
    ctx_seq:  [K, C, H, W]
    fut_seq:  [M, C, H, W]
    pred_seq: [M, C, H, W]
    """
    ctx_imgs  = _to_pil_list(ctx_seq)
    gt_imgs   = _to_pil_list(fut_seq)
    pred_imgs = _to_pil_list(pred_seq)

    K = len(ctx_imgs)
    M = len(gt_imgs)
    assert M == len(pred_imgs), "fut_seq и pred_seq должны иметь одинаковую длину"

    # размеры
    W, H = ctx_imgs[0].size  # PIL size = (W, H)
    spacer_w = max(10, W // 20)  # вертикальный разделитель
    label_h  = max(16, H // 16)  # место под подпись

    GUTTER_W = 2  # белые разделители между всеми соседними кадрами


    # холст: колонки = K + 1(spacer) + M; ряды = 2 (у каждого тайла есть подпись)
    canvas_w = (
        K * W + max(0, K - 1) * GUTTER_W   # входы + промежутки между ними
        + spacer_w                         # большой белый разделитель
        + M * W + max(0, M - 1) * GUTTER_W # GT/Pred + промежутки между ними
    )
    canvas_h = 2 * (H + label_h)           # ДВА ряда, без доп. зазора
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # PIL подберёт базовый

    # функции размещения
    def paste_with_label(img, col, row, label: str):
        if col == K:
            return
        if col < K:
            x = col * W + max(0, col) * GUTTER_W
        else:
            right_idx = col - (K + 1)
            left_block_w = K * W + max(0, K - 1) * GUTTER_W
            x = left_block_w + spacer_w + right_idx * W + max(0, right_idx) * GUTTER_W
    
        y = row * (H + label_h)  # без горизонтального разделителя между рядами
    
        canvas.paste(img, (x, y))
        tw = draw.textlength(label, font=font)
        tx = x + (W - tw) / 2
        ty = y + H + max(0, (label_h - 12) // 2)
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

    # 1-й ряд: входы -> разделитель -> GT
    # входы: номера 1..K
    for i, img in enumerate(ctx_imgs):
        paste_with_label(img, i, 0, f"frame {i+1}")

    # разделитель (вертикальная полоса)
    x_sep0 = K * W + max(0, K - 1) * GUTTER_W
    x_sep1 = x_sep0 + spacer_w - 1
    draw.rectangle([x_sep0, 0, x_sep1, canvas_h], fill=(255, 255, 255))

    # GT: номера продолжаются K+1..K+M
    for j, img in enumerate(gt_imgs):
        col = K + 1 + j
        num = K + 1 + j
        paste_with_label(img, col, 0, f"frame {num}")

    # 2-й ряд: под GT — предсказания (те же номера), под входами — пусто
    for j, img in enumerate(pred_imgs):
        col = K + 1 + j
        num = K + 1 + j
        paste_with_label(img, col, 1, f"pred frame {num}")

    # в TensorBoard и файл
    grid_tensor = ToTensor()(canvas)  # [C, H, W] в [0,1]
    writer.add_image("Sequences/rollout/train", grid_tensor, global_step=iter_)
    os.makedirs(dir_imgs, exist_ok=True)
    torchvision.utils.save_image(grid_tensor, os.path.join(dir_imgs, f"rollout_{mode}_{iter_:06d}.png"))
