import json
import sys
from pathlib import Path

import numpy as np
import torch

project = Path(r'C:\Users\莫名的wsr\Desktop\论文台西\深度学习承载力\CNN与transform')
code_dir = project / r'04_代码\trafficability_dl'
sys.path.insert(0, str(code_dir))
from unet import UNet
from segformer import SegFormer

old_art = project / r'03_数据与模型\模型权重与训练产物'
corr_art = project / r'03_数据与模型\第五轮校正版\m5_corrected'
corr_models = corr_art / 'common_models'
npz = np.load(old_art / 'common_domain.npz')
y_mech_corr = np.load(corr_art / 'Y_mech_corrected.npy')
idx = 247
r, c = 7, 100
x = torch.from_numpy(npz['X'][idx:idx+1].astype('float32'))
gr, gc = int(npz['GR'][idx]), int(npz['GC'][idx])
code = (gr * 73856093) ^ (gc * 19349663)
print('tile', gr, gc, 'split', 'validation' if abs(code) % 100 < 30 else 'training')

def segformer_model():
    return SegFormer(13, 3, dims=(32, 64, 160, 256), heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1), decoder_embed=256, ffn_ratio=4)

runs = [
    ('ahp', old_art / 'common_ahp_unet.pth', UNet(13, 3), npz['Y_ahp'][idx, r, c]),
    ('ahp', old_art / 'common_ahp_segformer.pth', segformer_model(), npz['Y_ahp'][idx, r, c]),
    ('corrected_mech', corr_models / 'common_mech_corrected_unet.pth', UNet(13, 3), y_mech_corr[idx, r, c]),
    ('corrected_mech', corr_models / 'common_mech_corrected_segformer.pth', segformer_model(), y_mech_corr[idx, r, c]),
]
results = []
for tag, weight_path, model, target in runs:
    state = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0, :, r, c].numpy()
    result = {
        'label': tag,
        'weight_file': weight_path.name,
        'target_class': int(target),
        'predicted_class': int(probs.argmax()),
        'probabilities': [round(float(v), 6) for v in probs],
    }
    results.append(result)
    print(tag, weight_path.name, 'target', int(target), 'pred', int(probs.argmax()), 'probs', ','.join(f'{v:.6f}' for v in probs))

output = {
    'well_name': 'Dongyuan',
    'station_id': 'DHW-08',
    'common_domain_patch_index': idx,
    'tile_grid_row': gr,
    'tile_grid_column': gc,
    'within_tile_row': r,
    'within_tile_column': c,
    'split': 'validation' if abs(code) % 100 < 30 else 'training',
    'results': results,
}
out_path = project / r'03_数据与模型\borehole_proxy_validation_20261117\dongyuan_model_check.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print('saved', out_path)
