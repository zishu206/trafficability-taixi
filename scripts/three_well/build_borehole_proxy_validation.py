import json
from pathlib import Path

import numpy as np
import pandas as pd

project = Path(r'C:\Users\莫名的wsr\Desktop\论文台西\深度学习承载力\CNN与transform')
sample_dir = project / r'03_数据与模型\水文地质API小样_20261117'
out_dir = project / r'03_数据与模型\borehole_proxy_validation_20261117'
out_dir.mkdir(parents=True, exist_ok=True)

lith = pd.read_csv(sample_dir / 'lithology_intervals.csv')
logs = pd.read_csv(sample_dir / 'geophysical_logs.csv')
coverage = pd.read_csv(sample_dir / 'coverage_summary.csv')
hydro_dict = pd.read_csv(sample_dir / 'official_data_gov' / 'hydro_lithology_code_dictionary.csv')
geo_dict = pd.read_csv(sample_dir / 'official_data_gov' / 'geo_lithology_code_dictionary.csv')
geo_dict['geo_code'] = geo_dict['geo_code'].astype(str).str.zfill(3)

lith = lith.merge(hydro_dict[['hydro_code', 'geo_code', 'zh_label']], left_on='rock_code', right_on='hydro_code', how='left')
lith['numeric_code'] = lith['rock_code'].astype(str).str.extract(r'^(\d+)$')[0].str.zfill(3)
lith = lith.merge(geo_dict[['geo_code', 'zh_label', 'en_label']], left_on='numeric_code', right_on='geo_code', how='left', suffixes=('_hydro', '_geo'))
lith['official_zh'] = lith['zh_label_hydro'].combine_first(lith['zh_label_geo'])
lith['official_en'] = lith['en_label']
lith['official_geo_code'] = lith['geo_code_hydro'].combine_first(lith['geo_code_geo'])

joined_parts = []
for station, group in lith.groupby('station_id'):
    sub_logs = logs[logs['station_id'] == station].copy()
    sub_logs['rock_code_interval'] = None
    sub_logs['official_zh'] = None
    sub_logs['official_en'] = None
    sub_logs['official_geo_code'] = None
    for _, row in group.sort_values('top_depth_m').iterrows():
        mask = (sub_logs['depth_m'] >= row['top_depth_m']) & (sub_logs['depth_m'] < row['bottom_depth_m'])
        sub_logs.loc[mask, 'rock_code_interval'] = row['rock_code']
        sub_logs.loc[mask, 'official_zh'] = row['official_zh']
        sub_logs.loc[mask, 'official_en'] = row['official_en']
        sub_logs.loc[mask, 'official_geo_code'] = row['official_geo_code']
    joined_parts.append(sub_logs)
joined = pd.concat(joined_parts, ignore_index=True)

shallow = lith[lith['top_depth_m'] < 10].sort_values(['station_id', 'top_depth_m']).copy()
shallow_cols = ['station_id', 'well_name', 'top_depth_m', 'bottom_depth_m', 'thickness_m', 'rock_code', 'official_geo_code', 'official_zh', 'official_en']
shallow[shallow_cols].to_csv(out_dir / 'shallow_lithology_10m.csv', index=False, encoding='utf-8-sig')

medians = []
for depth_scope, data in [('0-10m', joined[joined['depth_m'] <= 10]), ('all', joined)]:
    summary = data.groupby(['station_id', 'well_name', 'rock_code_interval', 'official_zh', 'official_en'], dropna=False).agg(
        sample_count=('depth_m', 'count'),
        SP_median=('SP', 'median'),
        N16_median=('N16', 'median'),
        N64_median=('N64', 'median'),
        GAMMA_median=('GAMMA', 'median'),
    ).reset_index()
    summary.insert(0, 'depth_scope', depth_scope)
    medians.append(summary)
pd.concat(medians, ignore_index=True).to_csv(out_dir / 'log_lithology_medians.csv', index=False, encoding='utf-8-sig')

with open(out_dir / 'dongyuan_model_check.json', encoding='utf-8') as f:
    dongyuan_model = json.load(f)

# Existing project point labels for the three named boreholes.
point_path = project / r'03_数据与模型\第五轮校正版\m5_corrected\point_validation_corrected.csv'
points = pd.read_csv(point_path)
# Duplicate corrected columns are identical; retain first occurrence by position.
points = points.iloc[:, [0,1,2,5,6,7,8,9,15,16,17,18,19]]
points.columns = ['record_type','longitude','latitude','name','AHP_score','AHP_class','RCI_original','mech_original','well_depth','elevation','aquifer','RCI_corrected','mech_class_corrected']
named = points[points['name'].isin(['原斗', '明德', '東源'])].copy()
named.to_csv(out_dir / 'three_well_existing_reference_labels.csv', index=False, encoding='utf-8-sig')

summary_rows = []
for _, row in coverage.iterrows():
    station = row['station_id']
    point_row = named[named['name'] == row['well_name']].iloc[0]
    shallow_intervals = shallow[shallow['station_id'] == station]
    summary_rows.append({
        'station_id': station,
        'well_name': row['well_name'],
        'well_depth_m': row['well_depth_m'],
        'elevation_m': row['elevation_m'],
        'lithology_intervals': row['rock_interval_count'],
        'lithology_depth_range_m': f"{row['rock_min_top_depth_m']}-{row['rock_max_bottom_depth_m']}",
        'top_unrecorded_gap_m': row['rock_top_unrecorded_gap_m'],
        'internal_unrecorded_gap_m': row['rock_internal_unrecorded_gap_m'],
        'L_missing_intervals': row['explicit_L_interval_count'],
        'L_missing_thickness_m': row['explicit_L_thickness_m'],
        'geophysical_rows': row['phys_row_count'],
        'geophysical_depth_range_m': f"{row['phys_raw_min_depth_m']}-{row['phys_raw_max_depth_m']}" if pd.notna(row['phys_raw_min_depth_m']) else 'not available',
        'AHP_class': int(point_row['AHP_class']),
        'corrected_mech_class': int(point_row['mech_class_corrected']),
        'shallow_evidence': '; '.join(f"{a.rock_code}={a.official_zh} ({a.top_depth_m}-{a.bottom_depth_m} m)" for a in shallow_intervals.itertuples()),
        'permissible_validation': 'Lithology/grain-size and geophysical proxy consistency only; no CI, CBR, plate-bearing or vehicle-pass result',
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(out_dir / 'borehole_proxy_validation_summary.csv', index=False, encoding='utf-8-sig')

json_summary = {
    'evidence_level': 'supporting proxy validation',
    'total_lithology_intervals': int(len(lith)),
    'total_geophysical_log_rows': int(len(logs)),
    'named_boreholes': summary_rows,
    'dongyuan_model_check': dongyuan_model,
    'boundary_statements': [
        'The three boreholes were already present in the 888-record project compilation; the new evidence is depth-resolved lithology, grain-size codes and geophysical logs, not new independent locations.',
        'No borehole file contains CI, CBR, plate-bearing, vehicle load or observed pass/fail fields.',
        'Missing log values and code L are retained as missing; they are not interpolated as measured values.',
        'The AHP and corrected-mechanistic reference classes differ at all three wells; lithology and logs cannot determine which reference is mechanically true.',
        'The Dongyuan model check shows reproduction of the respective supervising reference on a validation tile, not independent mechanical validation.',
    ],
    'sources': {
        'hydrogeology_map': 'https://hydro.geologycloud.tw/map',
        'api_documentation': 'https://hydro.geologycloud.tw/swagger/api-docs/api',
        'plain_core_dataset': 'https://data.gov.tw/dataset/61131',
        'plain_well_log_dataset': 'https://data.gov.tw/dataset/61132',
        'licence': 'https://data.gov.tw/license?lang=zh-tw',
    },
}
with open(out_dir / 'borehole_proxy_validation.json', 'w', encoding='utf-8') as f:
    json.dump(json_summary, f, ensure_ascii=False, indent=2)

readme = '''# Borehole supporting proxy validation

This folder contains the deterministic three-well borehole proxy check used in the English manuscript. The evidence supports near-surface lithology, grain-size and geophysical consistency checks. It does not provide CI, CBR, plate-bearing, vehicle-load or observed pass/fail validation.

## Files

- `borehole_proxy_validation_summary.csv`: per-well coverage, missing intervals and existing reference labels.
- `shallow_lithology_10m.csv`: official depth-resolved lithology labels from 0 to 10 m.
- `log_lithology_medians.csv`: median SP, 16-inch/64-inch resistivity and Gamma values by lithology; missing values remain missing.
- `three_well_existing_reference_labels.csv`: AHP and corrected-mechanistic labels from the existing 888-record compilation.
- `dongyuan_model_check.json`: model-class and probability check at the Dongyuan validation tile.
- `borehole_proxy_validation.json`: machine-readable complete summary.

The three well locations were already included in the project compilation. The added value is depth-resolved evidence, not a new independent validation set.
'''
with open(out_dir / 'README_borehole_proxy_validation.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print(out_dir)
