import os
import argparse
import yaml
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.ad_r_jepa import AD_R_JEPA
from nuscenes_data import NuscData
from radar_ssl_data import RadarSSLData

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes

from utils.vox import Vox_util


###############################################
# SLURM detection
###############################################
def is_slurm_job():
    # SLURM setzt IMMER diese drei Variablen
    required = ["SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_JOB_NAME"]
    return all(v in os.environ for v in required)



###############################################
# BEVCar-style checkpoint directory
###############################################
def make_ckpt_dir(base_dir, exp_name, batch_size, lr):
    lrn = f"{lr:.1e}"
    lrn = lrn[0] + lrn[3:5] + lrn[-1]  # 3e-4 → 3e-4s
    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    name = f"{exp_name}_{batch_size}_{lrn}_{timestamp}"
    return os.path.join(base_dir, name)


###############################################
# Keep only latest N checkpoints
###############################################
def keep_latest_checkpoints(ckpt_dir, keep_latest):
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )
    if len(ckpts) > keep_latest:
        to_delete = ckpts[:len(ckpts) - keep_latest]
        for f in to_delete:
            os.remove(os.path.join(ckpt_dir, f))


###############################################
# Argument parser
###############################################
def parse_args():
    parser = argparse.ArgumentParser("Radar JEPA SSL Pretrain (BEVCar)")
    parser.add_argument("--cfg_file", type=str, default="configs/ad_r_jepa_nuscenes.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="output/ad_r_jepa_ssl")
    return parser.parse_args()


###############################################
# Dataloader
###############################################
def build_dataloader(cfg, batch_size, workers):

    data_cfg = cfg["DATA_CONFIG"]

    nusc = NuScenes(
        version=data_cfg["VERSION"],
        dataroot=data_cfg["DATASET_ROOT"],
        verbose=True
    )

    nusc_maps = {
        name: NuScenesMap(
            dataroot=data_cfg["DATASET_ROOT"],
            map_name=name
        )
        for name in [
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
            "singapore-onenorth",
        ]
    }

    bounds = (-50, 50, -50, 50, -5, 3)
    res_3d = (1, 200, 200)

    centroid = np.zeros((1, 3), dtype=np.float32)
    Z, Y, X = res_3d

    vox_util = Vox_util(
        Z,
        Y,
        X,
        scene_centroid=torch.from_numpy(centroid).float().cuda(),
        bounds=bounds,
        assert_cube=False
    )

    dataset = RadarSSLData(
        nusc=nusc,
        vox_util=vox_util,
        nsweeps=data_cfg["NSWEEPS"],
        Z=1,
        Y=200,
        X=200,
        use_shallow_metadata=data_cfg["USE_SHALLOW_METADATA"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


###############################################
# Model builder
###############################################
def build_model(cfg):
    model_cfg = cfg["MODEL"]
    grid_size = cfg["DATA_CONFIG"]["POINT_CLOUD_RANGE_GRID_SIZE"]
    voxel_size = cfg["DATA_CONFIG"]["VOXEL_SIZE"]
    point_cloud_range = cfg["DATA_CONFIG"]["POINT_CLOUD_RANGE"]

    model = AD_R_JEPA(
        model_cfg=model_cfg,
        grid_size=grid_size,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range
    )
    return model


###############################################
# MAIN TRAINING LOOP
###############################################
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    # YAML checkpoint config
    ckpt_base = cfg.get("ckpt_dir", "model_checkpoints")
    keep_latest = cfg.get("keep_latest", 10)
    init_dir = cfg.get("init_dir", "")
    exp_name = cfg.get("exp_name", "jepa_ssl")

    dataloader = build_dataloader(cfg, args.batch_size, args.workers)
    model = build_model(cfg)

    
    model = model.cuda()
    model.train()
    ###############################################
    # Parameter statistics
    ###############################################
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable parameters: {trainable_params:,}")

    non_trainable_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    print(f"Non-trainable parameters: {non_trainable_params:,}")

    total_params = trainable_params + non_trainable_params
    print(f"Total parameters (trainable + fixed): {total_params:,}")


    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ###############################################
    # SLURM checkpoint directory
    ###############################################
    mask_and_exp_name = f"Mask_{cfg['MODEL']['MASKED_RATIO']:.2f}_{exp_name}"
    print("Mask:")
    print(f"Mask_{cfg['MODEL']['MASKED_RATIO']:.2f}_{exp_name}")
    if is_slurm_job():
        ckpt_dir = make_ckpt_dir(
            ckpt_base,
            mask_and_exp_name,
            args.batch_size,
            args.lr
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[SLURM] Checkpoints werden gespeichert in: {ckpt_dir}")
    else:
        ckpt_dir = None
        print("[LOCAL] Keine Checkpoints (SLURM nicht aktiv)")
    



    ###############################################
    # Optional: Resume (not required now)
    ###############################################
    global_step = 0

    ###############################################
    # Training
    ###############################################
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for i, batch_dict in enumerate(dataloader):

            for k in batch_dict.keys():
                if isinstance(batch_dict[k], torch.Tensor):
                    batch_dict[k] = batch_dict[k].cuda(non_blocking=True)

            optimizer.zero_grad()

            batch_out = model(batch_dict)
            loss, tb_dict = model.get_loss()

            loss.backward()
            optimizer.step()

            model.update_target_encoder()

            epoch_loss += loss.item()
            global_step += 1

            if ckpt_dir is not None and global_step % 1000 == 0:

                ckpt_path = os.path.join(
                    ckpt_dir,
                    f"model-{global_step:09d}.pth"
                )

                torch.save(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "cfg": cfg,
                    },
                    ckpt_path
                )

                keep_latest_checkpoints(ckpt_dir, keep_latest)

                print(f"[CKPT] Saved {ckpt_path}")

            if i % 50 == 0:
                print(
                    f"Epoch {epoch} Iter {i} "
                    f"Loss {loss.item():.4f} "
                    f"JEPA {tb_dict['loss']:.4f} "
                    f"Mask {tb_dict['mask_ratio']:.4f}"
                )

        avg_loss = epoch_loss / len(dataloader)
        print(
            f"Epoch {epoch} done in {time.time() - start_time:.1f}s, "
            f"avg loss {avg_loss:.4f}"
        )




if __name__ == "__main__":
    main()
