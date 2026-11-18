from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JS_PATH = ROOT / 'js_inspect' / 'map.js'
OUT_DIR = ROOT / 'official_data_gov'
OUT_DIR.mkdir(parents=True, exist_ok=True)

text = JS_PATH.read_text(encoding='utf-8', errors='replace')
start = text.find('let rockTable={')
end = text.find('rockTable.grainToCh=', start)
if start < 0 or end < 0:
    raise RuntimeError('未能定位官方岩性映射代码段')
block = text[start:end]
pattern = re.compile(r'\["([^"]+)","([^"]+)","([^"]+)"\]')
triples = pattern.findall(block)

hydro_rows = []
geo_rows = []
seen_hydro = set()
seen_geo = set()
for first, second, third in triples:
    if first.isdigit():
        if first not in seen_geo:
            geo_rows.append({
                'geo_code': first,
                'zh_label': second,
                'en_label': third,
                'source': 'hydro.geologycloud.tw/map/js/map.js rockTable',
            })
            seen_geo.add(first)
    else:
        if first not in seen_hydro:
            hydro_rows.append({
                'hydro_code': first,
                'geo_code': second,
                'zh_label': third,
                'source': 'hydro.geologycloud.tw/map/js/map.js rockTable',
            })
            seen_hydro.add(first)

hydro_path = OUT_DIR / 'hydro_lithology_code_dictionary.csv'
geo_path = OUT_DIR / 'geo_lithology_code_dictionary.csv'
for path, rows, fieldnames in [
    (hydro_path, hydro_rows, ['hydro_code', 'geo_code', 'zh_label', 'source']),
    (geo_path, geo_rows, ['geo_code', 'zh_label', 'en_label', 'source']),
]:
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

summary = {
    'source_file': str(JS_PATH),
    'source_sha256': hashlib.sha256(JS_PATH.read_bytes()).hexdigest().upper(),
    'block_start': start,
    'block_end': end,
    'hydro_dictionary': {'path': str(hydro_path), 'rows': len(hydro_rows), 'columns': ['hydro_code', 'geo_code', 'zh_label', 'source']},
    'geo_dictionary': {'path': str(geo_path), 'rows': len(geo_rows), 'columns': ['geo_code', 'zh_label', 'en_label', 'source']},
    'key_codes': {
        'vfS': next((r for r in hydro_rows if r['hydro_code'] == 'vfS'), None),
        'ZfS': next((r for r in hydro_rows if r['hydro_code'] == 'ZfS'), None),
        'L': next((r for r in hydro_rows if r['hydro_code'] == 'L'), None),
    },
}
summary_path = OUT_DIR / 'js_code_dictionary_summary.json'
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
