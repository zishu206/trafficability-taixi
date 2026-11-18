import sys
from pathlib import Path

import numpy as np
import torch

code_dir = Path(r'C:\Users\莫名的wsr\Desktop\论文台西\深度学习承载力\CNN与transform\04_代码\trafficability_dl')
sys.path.insert(0, str(code_dir))
from unet import UNet
from segformer import SegFormer

artifact_dir = Path(r'C:\Users\莫名的wsr\Desktop\论文台西\深度学习承载力\CNN与transform\03_数据与模型\模型权重与训练产物')
npz = np.load(artifact_dir / 'common_domain.npz')
idx = 247
r, c = 7, 100
x = torch.from_numpy(npz['X'][idx:idx+1].astype('float32'))

for tag in ['ahp', 'mech']:
    for model_name, model in [
        ('unet', UNet(13, 3)),
        ('segformer', SegFormer(13, 3, dims=(32, 64, 160, 256), heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1), decoder_embed=256, ffn_ratio=4)),
    ]:
        state = torch.load(artifact_dir / f'common_{tag}_{model_name}.pth', map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0, :, r, c].numpy()
        print(tag, model_name, 'pred_class', int(probs.argmax()), 'probs', ','.join(f'{v:.6f}' for v in probs))
print('reference labels at pixel: ahp', npz['Y_ahp'][idx, r, c], 'mech', npz['Y_mech'][idx, r, c])
