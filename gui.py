<<<<<<< HEAD
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.CamoFormer import CamoFormer

def parse_args():
    parser = argparse.ArgumentParser("FSPNet-Transformer")
    parser.add_argument('--base_lr', default=(1e-4), type=float, help='learning rate')
    parser.add_argument('--batch_size', default=6, type=int, help='batch size per GPU')
    parser.add_argument("--resume", default=None)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--path', type=str, default=r'E:\Mr.Wu\dataset\CodDataset', help='path to train dataset')
    parser.add_argument('--pretrain', type=str,
                        default=r'E:\Mr.Wu\codes\伪装目标检测软件\pretrain.pth',
                        help='path to pretrain model')

    # DDP configs:
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args

# 加载预训练模型
def load_model(model_path):
    args = parse_args()
    ### model ###
    device = torch.device('cuda:0')
    net = CamoFormer(cfg=None)
    key = torch.load(args.pretrain, map_location=device)
    net.load_state_dict(key['state_dict'], strict=True)
    net.eval()
    return net

# 进行推理
def segment_image(image, model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    W, H = image.size
    shape = H, W
    image = transform(image).unsqueeze(0).to(next(model.parameters()).device)
    image = F.interpolate(image, (384, 384), mode='bilinear')
    with torch.no_grad():
        _, _, _, _, output = model(image)
    out = F.interpolate(output, size=shape, mode='bilinear', align_corners=False)
    pred = out.sigmoid().data.cpu().numpy().squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # 标准化处理,把数值范围控制到(0,1)

    pred_image = Image.fromarray((pred * 255).astype(np.uint8))
    return pred_image

# 加载模型
model = load_model(r'E:\Mr.Wu\codes\FSPNet_Weak\checkpoint\newpretrain\P+CrossViT.pth')

def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize((300, 300))  # 调整图像尺寸
    img_tk = ImageTk.PhotoImage(img)

    # 在标签中显示上传的图像
    img_label.config(image=img_tk)
    img_label.image = img_tk  # 保持引用，避免被垃圾回收

    # 进行推理并显示结果
    prediction = segment_image(img, model)
    pred_tk = ImageTk.PhotoImage(prediction.resize((300, 300)))

    # 在标签中显示分割结果
    result_label.config(image=pred_tk)
    result_label.image = pred_tk  # 保持引用，避免被垃圾回收


# 创建主窗口
root = tk.Tk()
root.title("Image Segmentation")
root.geometry("1600x1600")  # 设置主窗口大小

# 创建标签和按钮的框架
frame = tk.Frame(root)
frame.pack(expand=True)

# 显示上传的图像
img_label = tk.Label(frame)
img_label.pack()

# 创建上传按钮
upload_button = tk.Button(frame, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_button.pack(pady=20)

# 显示分割结果
result_label = tk.Label(frame)
result_label.pack(pady=20)

# 运行主循环
=======
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.CamoFormer import CamoFormer

def parse_args():
    parser = argparse.ArgumentParser("FSPNet-Transformer")
    parser.add_argument('--base_lr', default=(1e-4), type=float, help='learning rate')
    parser.add_argument('--batch_size', default=6, type=int, help='batch size per GPU')
    parser.add_argument("--resume", default=None)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--path', type=str, default=r'E:\Mr.Wu\dataset\CodDataset', help='path to train dataset')
    parser.add_argument('--pretrain', type=str,
                        default=r'E:\Mr.Wu\codes\伪装目标检测软件\pretrain.pth',
                        help='path to pretrain model')

    # DDP configs:
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args

# 加载预训练模型
def load_model(model_path):
    args = parse_args()
    ### model ###
    device = torch.device('cuda:0')
    net = CamoFormer(cfg=None)
    key = torch.load(args.pretrain, map_location=device)
    net.load_state_dict(key['state_dict'], strict=True)
    net.eval()
    return net

# 进行推理
def segment_image(image, model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    W, H = image.size
    shape = H, W
    image = transform(image).unsqueeze(0).to(next(model.parameters()).device)
    image = F.interpolate(image, (384, 384), mode='bilinear')
    with torch.no_grad():
        _, _, _, _, output = model(image)
    out = F.interpolate(output, size=shape, mode='bilinear', align_corners=False)
    pred = out.sigmoid().data.cpu().numpy().squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # 标准化处理,把数值范围控制到(0,1)

    pred_image = Image.fromarray((pred * 255).astype(np.uint8))
    return pred_image

# 加载模型
model = load_model(r'E:\Mr.Wu\codes\FSPNet_Weak\checkpoint\newpretrain\P+CrossViT.pth')

def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize((300, 300))  # 调整图像尺寸
    img_tk = ImageTk.PhotoImage(img)

    # 在标签中显示上传的图像
    img_label.config(image=img_tk)
    img_label.image = img_tk  # 保持引用，避免被垃圾回收

    # 进行推理并显示结果
    prediction = segment_image(img, model)
    pred_tk = ImageTk.PhotoImage(prediction.resize((300, 300)))

    # 在标签中显示分割结果
    result_label.config(image=pred_tk)
    result_label.image = pred_tk  # 保持引用，避免被垃圾回收


# 创建主窗口
root = tk.Tk()
root.title("Image Segmentation")
root.geometry("1600x1600")  # 设置主窗口大小

# 创建标签和按钮的框架
frame = tk.Frame(root)
frame.pack(expand=True)

# 显示上传的图像
img_label = tk.Label(frame)
img_label.pack()

# 创建上传按钮
upload_button = tk.Button(frame, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_button.pack(pady=20)

# 显示分割结果
result_label = tk.Label(frame)
result_label.pack(pady=20)

# 运行主循环
>>>>>>> 001be03 (first commit)
root.mainloop()