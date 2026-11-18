from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SAMPLE_ROOT = ROOT.parent
INTERVALS = SAMPLE_ROOT / 'lithology_intervals.csv'
HYDRO_DICT = ROOT / 'hydro_lithology_code_dictionary.csv'
GEO_DICT = ROOT / 'geo_lithology_code_dictionary.csv'
OUT = ROOT / 'three_well_official_labels.csv'

intervals = pd.read_csv(INTERVALS, dtype=str, encoding='utf-8-sig')
hydro = pd.read_csv(HYDRO_DICT, dtype=str, encoding='utf-8-sig')
geo = pd.read_csv(GEO_DICT, dtype=str, encoding='utf-8-sig')
intervals['thickness_m'] = pd.to_numeric(intervals['thickness_m'], errors='coerce')

rows = []
for (domain, raw_code), part in intervals.groupby(['source_region', 'rock_code'], dropna=False):
    if domain == 'P':
        normalized = raw_code
        match = hydro[hydro['hydro_code'] == raw_code]
        if match.empty:
            zh = ''
            geo_code = ''
        else:
            zh = match.iloc[0]['zh_label']
            geo_code = match.iloc[0]['geo_code']
    else:
        normalized = str(raw_code).zfill(3) if str(raw_code).isdigit() else raw_code
        geo_code = normalized
        match = geo[geo['geo_code'] == normalized]
        zh = match.iloc[0]['zh_label'] if not match.empty else ''
    geo_match = geo[geo['geo_code'] == geo_code]
    en = geo_match.iloc[0]['en_label'] if not geo_match.empty else ''
    rows.append({
        'domain': domain,
        'raw_code': raw_code,
        'normalized_code': normalized,
        'geo_code': geo_code,
        'official_zh_label': zh,
        'official_en_label': en,
        'interval_count': len(part),
        'total_thickness_m': round(part['thickness_m'].sum(), 3),
        'wells': ';'.join(sorted(part['station_id'].unique())),
        'evidence_note': '官方前端脚本映射；仅表示岩性/岩类，不表示CI、CBR、承载力或车辆通行结论',
    })

labels = pd.DataFrame(rows).sort_values(['domain', 'normalized_code'])
labels.to_csv(OUT, index=False, encoding='utf-8-sig')
print(OUT)
print(labels.to_string(index=False))
