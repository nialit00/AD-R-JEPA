import torch
from utils import parser
from utils import saverloader
from models import model_utils
from datasets.nuscenes_data import NuScenesDataset
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 1) Config laden
# ---------------------------------------------------------
cfg = parser.load_cfg('configs/bevcar/bevcar_1x5.yaml')

# ---------------------------------------------------------
# 2) Dataset + ein Batch holen
# ---------------------------------------------------------
dataset = NuScenesDataset(cfg, 'trainval')
loader = DataLoader(dataset, batch_size=1, shuffle=False)
sample = next(iter(loader))

# ---------------------------------------------------------
# 3) BEVCar Scratch Modell
# ---------------------------------------------------------
model_scratch = model_utils.build_model(cfg)
model_scratch = torch.nn.DataParallel(model_scratch).cuda()
model_scratch.eval()

with torch.no_grad():
    feats_scratch = model_scratch.module.image_encoder(sample['imgs'].cuda())

# ---------------------------------------------------------
# 4) BEVCar mit JEPA init
# ---------------------------------------------------------
model_jepa = model_utils.build_model(cfg)
model_jepa = torch.nn.DataParallel(model_jepa).cuda()

# JEPA Checkpoint laden
saverloader.load('PATH/TO/JEPA_CKPT.pth', model_jepa)

model_jepa.eval()
with torch.no_grad():
    feats_jepa = model_jepa.module.image_encoder(sample['imgs'].cuda())

# ---------------------------------------------------------
# 5) Vergleich
# ---------------------------------------------------------
diff = torch.mean(torch.abs(feats_scratch - feats_jepa))
print("Feature-Differenz:", diff.item())
