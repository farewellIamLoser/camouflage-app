from torch.utils.data import DataLoader
import os
import builtins
import argparse
import torch
import torch.distributed as dist
import time
import utils.metrics as Measure
import torch.nn.functional as F
import numpy as np
import dataset
import loss
from model.TestCamoFormer import CamoFormer
from utils.tools import *
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser("FSPNet-Transformer")
    parser.add_argument('--base_lr', default=(1e-4), type=float, help='learning rate')
    parser.add_argument('--batch_size', default=6, type=int, help='batch size per GPU')
    parser.add_argument("--resume", default=None)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--path', type=str, default=r'E:\Mr.Wu\dataset\CodDataset', help='path to train dataset')
    parser.add_argument('--pretrain', type=str,
                        default=r'E:\Mr.Wu\codes\FSPNet_Weak\checkpoint\newpretrain\module_ablation\P+CrossViT.pth',
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


def get_transform(ops=[0, 1, 2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op == 0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op == 1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op == 2:
        pp = Crop(0.7, 0.7)
    return pp


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)


def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
            print("args.rank = {}; args.gpu = {}".format(args.rank, args.gpu))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    ### model ###
    net = CamoFormer(cfg=None)
    if args.pretrain:
        key = torch.load(args.pretrain)
        net.load_state_dict(key['state_dict'], strict=True)
    if args.distributed:

        os.system("nvidia-smi")
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            net = net.cuda(args.gpu)
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
    else:
        net.cuda()

    encoder_param = []
    decoer_param = []
    for name, param in net.named_parameters():
        if "encoder" in name:
            encoder_param.append(param)
        else:
            decoer_param.append(param)
    TestDir = [args.path + r'\TestDataset\CAMO',
               args.path + r'\TestDataset\COD10K', args.path + r'\TestDataset\N4CK']
    # TestDir = [args.path + r'\TestDataset\COD10K']
    TestDataset = [dataset.TestDataset(v, size=384) for v in TestDir]
    TestDataloaders = [DataLoader(v, batch_size=1, num_workers=1) for v in TestDataset]
    ### main loop ###
    for curr_epoch in range(0, 4):

        maes = [1, 1, 1, 1]
        for i, TestDataloader in enumerate(TestDataloaders):
            net.train(False)
            FM = Measure.Fmeasure()
            WFM = Measure.WeightedFmeasure()
            SM = Measure.Smeasure()
            EM = Measure.Emeasure()
            MAE = Measure.MAE()
            with torch.no_grad():
                st = time.time()
                for data in TestDataloader:
                    image, mask, name, shape = data['img'].cuda(args.rank), data['label'], data[
                        'name'], \
                        data['shape']
                    _, _, _, _, out = net(image)
                    out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
                    pred = out.sigmoid().data.cpu().numpy().squeeze()
                    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # 标准化处理,把数值范围控制到(0,1)
                    mask = mask.numpy().astype(np.float32).squeeze()
                    mask /= (mask.max() + 1e-8)

                    pred_image = Image.fromarray((pred * 255).astype(np.uint8))
                    pred_image.save(name[0].replace('GT', 'result'))

                    FM.step(pred=pred * 255, gt=mask * 255)
                    WFM.step(pred=pred* 255, gt=mask* 255)
                    SM.step(pred=pred* 255, gt=mask* 255)
                    EM.step(pred=pred* 255, gt=mask* 255)
                    MAE.step(pred=pred* 255, gt=mask* 255)
                    fm = FM.get_results()["fm"]
                    wfm = WFM.get_results()["wfm"]
                    sm = SM.get_results()["sm"]
                    em = EM.get_results()["em"]
                    mae = MAE.get_results()["mae"]

                results = {
                    "MAE": mae,
                    "adpFm": fm["adp"],
                    "meanEm": em["curve"].mean(),
                    "Smeasure": sm,
                }
                print(results)
                print('dataset: {}, spend_time:{}'.format(
                    TestDataloader.dataset.data_name, (time.time() - st)))
                maes[i] = mae
            net.train(True)

if __name__ == '__main__':
    args = parse_args()
    main(args)