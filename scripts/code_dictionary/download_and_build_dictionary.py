from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / 'raw'
RAW.mkdir(parents=True, exist_ok=True)

SOURCES = [
    {
        'dataset_id': '16799',
        'dataset_title': '水文地質鑽探資料庫',
        'url': 'https://www.gsmma.gov.tw/uploads/17503844515568fWuXe2C.csv',
        'quality_url': 'https://quality.data.gov.tw/dq_download_csv.php?nid=16799&md5_url=92fee6fc9ff144b35e7adad2475e4e9d',
        'raw_name': '16799_水文地质钻探资料库.csv',
    },
    {
        'dataset_id': '61131',
        'dataset_title': '平原區水文地質鑽探-鑽孔岩心紀錄',
        'url': 'https://www.gsmma.gov.tw/uploads/1718781576486EoaffVM1.csv',
        'quality_url': 'https://quality.data.gov.tw/dq_download_csv.php?nid=61131&md5_url=6ac73516ed97cfdc97f9018f94d66a87',
        'raw_name': '61131_平原钻孔岩心记录.csv',
    },
    {
        'dataset_id': '61132',
        'dataset_title': '平原區水文地質鑽探-鑽孔井錄',
        'url': 'https://www.gsmma.gov.tw/uploads/1697249779106gZCOirdl.CSV',
        'quality_url': 'https://quality.data.gov.tw/dq_download_csv.php?nid=61132&md5_url=13296cf3db708653296ff73e92e02443',
        'raw_name': '61132_平原钻孔井录.csv',
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def download(source: dict) -> Path:
    target = RAW / source['raw_name']
    if not target.exists():
        req = Request(
            source['quality_url'],
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
                'Accept': 'text/csv,application/csv,text/plain,*/*;q=0.8',
                'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
                'Referer': 'https://data.gov.tw/',
            },
        )
        with urlopen(req, timeout=120) as response:
            data = response.read()
        if data[:1].lower() == b'<':
            raise ValueError(f'下载内容疑似HTML而非CSV: {source["quality_url"]}')
        target.write_bytes(data)
    return target


def read_csv_loose(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ['utf-8-sig', 'utf-8', 'cp950']:
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def main():
    manifest = []
    frames = {}
    for source in SOURCES:
        path = download(source)
        df = read_csv_loose(path)
        frames[source['dataset_id']] = df
        manifest.append({
            **source,
            'raw_path': str(path),
            'rows': len(df),
            'columns': len(df.columns),
            'sha256': sha256(path),
            'license': '政府資料開放授權條款-第1版',
            'charge': '免費',
        })

    lith = frames['61131']
    expected = ['縣市代碼', '行政區域代碼', '井名', '井號', '深度上限', '深度下限', '岩性代碼']
    missing = [c for c in expected if c not in lith.columns]
    if missing:
        raise ValueError(f'岩心记录缺少字段: {missing}; actual={lith.columns.tolist()}')
    label_cols = [c for c in ['岩類代碼說明', '岩類簡稱', '岩類說明'] if c in lith.columns]
    dictionary_rows = []
    for code, part in lith.groupby('岩性代碼', dropna=False, sort=True):
        row = {
            '岩性代碼': code,
            '层段记录数': len(part),
            '涉及井号数': part['井號'].nunique(),
            '最小深度上限': pd.to_numeric(part['深度上限'], errors='coerce').min(),
            '最大深度下限': pd.to_numeric(part['深度下限'], errors='coerce').max(),
        }
        for col in label_cols:
            nonblank = part[part[col] != '']
            row[col] = nonblank.iloc[0][col] if len(nonblank) else ''
            row[col + '_非空记录数'] = len(nonblank)
        dictionary_rows.append(row)
    dictionary = pd.DataFrame(dictionary_rows)
    dictionary_path = BASE / 'plain_lithology_code_dictionary.csv'
    dictionary.to_csv(dictionary_path, index=False, encoding='utf-8-sig')

    logs = frames['61132']
    basic = frames['16799']
    summary = {
        'sources': manifest,
        'basic_metadata_shape': [len(basic), len(basic)],
        'basic_columns': basic.columns.tolist(),
        'plain_lithology_shape': [len(lith), len(lith)],
        'plain_lithology_columns': lith.columns.tolist(),
        'plain_logs_shape': [len(logs), len(logs)],
        'plain_logs_columns': logs.columns.tolist(),
        'code_dictionary_rows': len(dictionary),
        'code_dictionary_path': str(dictionary_path),
        'license_basis': 'data.gov.tw页面载明政府資料開放授權條款-第1版，免费；具体条款仍需以平台最新页面为准。',
    }
    (BASE / 'official_data_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
