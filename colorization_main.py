import sys,argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from colorization_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import time
import torch
from share import *
from config import cfg



def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="continue train from resume",
    )
    parser.add_argument(
        "-m",
        "--multicolor",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="continue train from resume",
    )
    parser.add_argument(
        "-s",
        "--usesam",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="continue train from resume",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args,_ = get_parser()

    if args.train:
        n_gpu = cfg.n_gpu
        init_model_path = cfg.init_model_path

        batch_size = cfg.batch_size
        logger_freq = cfg.logger_freq
        learning_rate = cfg.learning_rate_multiplier * n_gpu
        sd_locked = cfg.sd_locked
        only_mid_control = cfg.only_mid_control

        model = create_model(cfg.cldm_v15_config).cpu()

        model.load_state_dict(load_state_dict(init_model_path, location='cpu'))
        model.learning_rate = learning_rate
        model.sd_locked = sd_locked
        model.only_mid_control = only_mid_control

        dataset = MyDataset(img_dir=cfg.coco_img_dir, caption_dir=cfg.coco_caption_dir)

        dataloader = DataLoader(dataset, num_workers=cfg.num_workers, batch_size=batch_size, shuffle=True)
        logger = ImageLogger(batch_frequency=logger_freq)
        trainer = pl.Trainer(gpus=n_gpu, precision=cfg.precision, callbacks=[logger])
        # Train!
        trainer.fit(model, dataloader)

    else: # test or val

        resume_path = cfg.resume_checkpoint

        batch_size = cfg.test_batch_size

        model = create_model(cfg.cldm_v15_config).cpu()
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))

        trainer = pl.Trainer(gpus=cfg.test_n_gpu, precision=cfg.test_precision)
        if args.multicolor: # test demo
            if args.usesam: # -m -s
                model.usesam = True
                dataset = MyDataset(img_dir=cfg.example_img_dir, caption_dir=cfg.sam_caption_dir, split='test', use_sam=True)
                dataloader = DataLoader(dataset, num_workers=cfg.test_num_workers, batch_size=batch_size, shuffle=False)
                trainer.test(model, dataloader)
            else: # -m
                model.usesam = False
                dataset = MyDataset(img_dir=cfg.example_img_dir, caption_dir=cfg.example_caption_dir, split='test')
                dataloader = DataLoader(dataset, num_workers=cfg.test_num_workers, batch_size=batch_size, shuffle=False)
                trainer.test(model, dataloader)
        else: # val
            model.usesam = False
            dataset = MyDataset(img_dir=cfg.coco_img_dir, caption_dir=cfg.coco_caption_dir, split='val')
            dataloader = DataLoader(dataset, num_workers=cfg.test_num_workers, batch_size=batch_size, shuffle=False)
            trainer.test(model, dataloader)

