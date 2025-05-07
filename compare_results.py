import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from data.data import get_eval_set, get_CEC_eval_set
from net.CIDNet import CIDNet
from data.options import option
from tqdm import tqdm
from torchvision.transforms.functional import resize, pad, to_tensor
from glob import glob

opt = option().parse_args()

def resize_or_pad(img_tensor, target_multiple=16):
    """ 將影像 resize 或 padding 成為指定倍數，避免尺寸不一致 """
    _, _, h, w = img_tensor.shape
    new_h = (h + target_multiple - 1) // target_multiple * target_multiple
    new_w = (w + target_multiple - 1) // target_multiple * target_multiple
    pad_h = new_h - h
    pad_w = new_w - w
    # 使用 F.pad，格式為 (left, right, top, bottom)
    padded_img = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded_img, (pad_h, pad_w)


# Load model
def load_model(weight_path):
    model = CIDNet().cuda()
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

# Concatenate three images horizontally
def concat_images(img1, img2, img3):
    w, h = img1.size
    new_img = Image.new('RGB', (w * 3, h))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (w, 0))
    new_img.paste(img3, (w * 2, 0))
    return new_img

# Get test loader
def get_test_loader():
    test_set = get_CEC_eval_set(opt.data_val_CEC)

    loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return loader

if __name__ == '__main__':
    n_epoch = 80
    # n_epoch = opt.start_epoch if opt.start_epoch > 0 else opt.nEpochs
    weight_path = f"./weights/train/epoch_{n_epoch}.pth"
        # Create directory for results
    save_dir = f"./results/comparisons_{n_epoch}_epoch"
    os.makedirs(save_dir, exist_ok=True)
    model = load_model(weight_path)
    test_loader = get_test_loader()

    to_pil = transforms.ToPILImage()

    for idx, batch in enumerate(tqdm(test_loader)):
        if len(batch) == 2:
            input_img, gt_img = batch
            input_path, gt_path = None, None
        else:
            input_img, gt_img, input_path, gt_path = batch[0], batch[1], batch[2][0], batch[3][0]

        input_img = input_img.cuda()
        input_img, (pad_h, pad_w) = resize_or_pad(input_img)

        with torch.no_grad():
            output_img = model(input_img)
            output_img = output_img[:, :, :output_img.shape[2]-pad_h, :output_img.shape[3]-pad_w]

        input_pil = to_pil(input_img.squeeze(0).cpu().clamp(0, 1)[:, :input_img.shape[2]-pad_h, :input_img.shape[3]-pad_w])
        # gt_img = gt_img.cuda() if isinstance(gt_img, torch.Tensor) else gt_img[0].cuda()
        if isinstance(gt_img, (tuple, list)):
            gt_img = gt_img[0]
        
        # 處理 ground truth image
        if isinstance(gt_img, str):
            if not os.path.isfile(gt_img):
                print(f"[WARNING] Sample {idx}: GT path '{gt_img}' not found. Skipping.")
                continue
            if gt_img is None or not os.path.isfile(gt_img):
                raise FileNotFoundError(f"[ERROR] Ground truth image path not found: {gt_img}")
            gt_img = Image.open(gt_img).convert("RGB")
            gt_img = to_tensor(gt_img).unsqueeze(0).cuda()
        elif isinstance(gt_img, Image.Image):
            gt_img = to_tensor(gt_img).unsqueeze(0).cuda()
        elif isinstance(gt_img, torch.Tensor):
            gt_img = gt_img.cuda()
        else:
            raise TypeError(f"[ERROR] Unexpected gt_img type: {type(gt_img)}")
        gt_pil = to_pil(gt_img.squeeze(0).cpu().clamp(0, 1))
        output_pil = to_pil(output_img.squeeze(0).cpu().clamp(0, 1))

        comparison = concat_images(input_pil, gt_pil, output_pil)

        filename = (
            os.path.basename(input_path).replace(".png", "_compare.png")
            if input_path
            else f"{idx:04d}_compare.png"
        )
        comparison.save(os.path.join(save_dir, filename))


    print(f"[INFO] Comparison images saved to {save_dir}")
    print(weight_path)



    output_dir = f"./results/comparisons_label_{n_epoch}"
    img_paths = sorted(glob(os.path.join(save_dir, "*_compare.png")))
    os.makedirs(output_dir, exist_ok=True)

    labels = ["Input", "Ground Truth", "Corrected"]
    
    # 嘗試載入字體
    try:
        font = ImageFont.truetype("arial.ttf", 40)
        draw.text((10, height - 50), metric_text, fill="white", font=font)
    except:
        font = ImageFont.load_default()
    
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        section_width = width // 3
    
        draw = ImageDraw.Draw(image)
    
        for i, label in enumerate(labels):
            x = section_width * i + section_width // 2 - 40  # 大致置中
            draw.text((x, 10), label, fill="white", font=font)
    
        # 儲存帶標籤圖片
        filename = os.path.basename(img_path)
        image.save(os.path.join(output_dir, filename))
    
    print(f"[INFO] 標註完成，共處理 {len(img_paths)} 張圖，輸出至 {output_dir}")