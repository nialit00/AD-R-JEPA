# AD-R-JEPA with BEV-MAE's masking

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import random

from nets.voxelnet import VoxelNet

from utils.jepa_voxel_adapter import JEPAVoxelAdapter


"""
class Encoder(nn.Module):
    
    #Radar-Encoder für JEPA:
    #nutzt BEVCar's VoxelNet direkt.
    
    def __init__(self, model_cfg, **kwargs):
        super().__init__()

        # VoxelNet ist der Radar-Backbone von BEVCar
        self.voxelnet = VoxelNet(
            use_col=False,
            reduced_zx=False,
            output_dim=128,
            use_radar_occupancy_map=False
        )

    def forward(self, voxel_features, voxel_coords, num_voxels):
        
        #Erwartet exakt die drei Tensoren aus BEVCar:
        #- voxel_features: (B, K, T, C)
        #- voxel_coords:   (B, K, 3)
        #- num_voxels:     (B,)
        
        bev = self.voxelnet(
            voxel_features=voxel_features,
            voxel_coords=voxel_coords,
            number_of_occupied_voxels=num_voxels,
            dinovoxel=None
        )
        # VoxelNet gibt (B, 128, Z=200, X=200)
        return bev
"""


        
class Predictor(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class AD_R_JEPA(nn.Module):
    """
    Radar JEPA (BEVCar-kompatibel)
    - Encoder = VoxelNet
    - Masking = BEV-MAE Style
    - Predictor = 128-Kanal CNN
    """

    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.masked_ratio = model_cfg["MASKED_RATIO"]
        self.voxel_adapter = JEPAVoxelAdapter()

        # -----------------------------
        # Encoder = BEVCar VoxelNet
        # -----------------------------
        self.radar_encoder = VoxelNet(
            use_col=False,
            reduced_zx=False,
            output_dim=128,
            use_radar_occupancy_map=False
        )

        self.radar_target_encoder = copy.deepcopy(self.radar_encoder)
        for p in self.radar_target_encoder.parameters():
            p.requires_grad = False



        # -----------------------------
        # Predictor (128 Kanäle)
        # -----------------------------
        self.predictor = Predictor(channels=128)

        # -----------------------------
        # Mask Tokens (128 Kanäle)
        # -----------------------------
        self.mask_token = nn.Parameter(torch.randn(1, 128, 1, 1))
        self.empty_token = nn.Parameter(torch.randn(1, 128, 1, 1))

        # -----------------------------
        # BEV Grid (200x200)
        # -----------------------------
        self.bev_x_shape = 200
        self.bev_y_shape = 200

        self.forward_re_dict = {}
        
        self.alpha = model_cfg["ALPHA"]
        self.beta = model_cfg["BETA"]
        
    @torch.no_grad()
    def update_target_encoder(self, m=0.996):
        """
        EMA-Update: target_encoder folgt langsam dem encoder.
        m ist das Momentum (z.B. 0.996).
        """
        for p, q in zip(self.radar_encoder.parameters(), self.radar_target_encoder.parameters()):
            q.data.mul_(m).add_(p.data, alpha=1 - m)

    # ---------------------------------------------------------
    # LOSS
    # ---------------------------------------------------------
    def get_loss(self):
        """
        JEPA loss: cosine similarity on masked BEV patches
        """

        pred = self.forward_re_dict['prediction']   # (B, C, H, W)
        tgt  = self.forward_re_dict['target']       # (B, C, H, W)
        mask = self.forward_re_dict['mask']         # (B, 1, H, W)

        # Wir vergleichen NUR die maskierten Zellen
        mask = (~mask).float()                      # target-maskierte Zellen = 1

        # Cosine similarity pro Zelle
        cos = F.cosine_similarity(pred, tgt, dim=1) # (B, H, W)

        # Loss = -cosine auf maskierten Zellen
        loss = -(cos * mask.squeeze(1)).sum() / (mask.sum() + 1e-6)

        tb_dict = {
            'loss': loss.item(),
            'mask_ratio': mask.mean().item()
        }

        return loss, tb_dict



    # ---------------------------------------------------------
    # FORWARD
    # ---------------------------------------------------------
    def forward(self, batch_dict):
        """
        JEPA forward pass for BEVCar Radar (korrekte Version)
        """

        # -----------------------------
        # 1. Original BEVCar Voxel-Inputs
        # -----------------------------
        voxel_features = batch_dict['voxel_input_feature_buffer']  # (B, K, T, 8)
        voxel_features = voxel_features[..., :7]                   # FIX: VoxelNet expects 7 features

        voxel_coords   = batch_dict['voxel_coordinate_buffer']
        num_voxels     = batch_dict['number_of_occupied_voxels']

        B = voxel_features.shape[0]

        # -----------------------------
        # 2. Encode mit VoxelNet
        # -----------------------------
        bev_ctx = self.radar_encoder(voxel_features, voxel_coords, num_voxels)        # (B, 128, H, W)
        bev_tgt = self.radar_target_encoder(voxel_features, voxel_coords, num_voxels) # (B, 128, H, W)

        B, C, H, W = bev_ctx.shape
        N = H * W

        # -----------------------------
        # 3. JEPA-Maskierung auf BEV-Grid
        # -----------------------------
        mask = (torch.rand(B, N, device=bev_ctx.device) > self.masked_ratio)
        mask = mask.view(B, 1, H, W)

        ctx = torch.where(mask, bev_ctx, self.mask_token)
        tgt = torch.where(~mask, bev_tgt, self.mask_token)

        # -----------------------------
        # 4. Normalisierung
        # -----------------------------
        ctx = F.normalize(ctx, dim=1)
        tgt = F.normalize(tgt, dim=1)

        # -----------------------------
        # 5. Predictor
        # -----------------------------
        pred = self.predictor(ctx)
        pred = F.normalize(pred, dim=1)

        # -----------------------------
        # 6. Outputs für Loss
        # -----------------------------

        self.forward_re_dict = {
            'context': ctx,
            'prediction': pred,
            'target': tgt,
            'mask': mask,

            # (B, H, W)
            'bev_mask_encoder': mask.squeeze(1),
            'bev_mask_target_encoder': (~mask).squeeze(1),

            # alte JEPA-Kompatibilität: immer False
            'bev_mask_encoder_empty': torch.zeros_like(mask.squeeze(1), dtype=torch.bool)
        }



        return batch_dict

   


