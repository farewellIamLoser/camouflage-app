python -m torch.distributed.launch --nproc_per_node=8 train.py --path "/path_to_dataset/data/COD10K/TrainDataset" --pretrain "/path_to_preTrainModel/preTrainModel/base_patch16_384.pth"