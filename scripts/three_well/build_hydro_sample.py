from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
DOWNLOADS = Path.home() / "Downloads"
ACCESS_DATE = "2026-11-17"
API_DOC = "https://hydro.geologycloud.tw/swagger/api-docs/api"
MAP_URL = "https://hydro.geologycloud.tw/map"
HELP_URL = "https://hydro.geologycloud.tw/help/help.html"
EXISTING_CSV = BASE.parent / "第五轮校正版" / "m5_corrected" / "point_validation_corrected.csv"

TARGETS = {
    "091803G1": {"name": "明德", "region": "P", "rock": "091803G1_rockrecord.json", "phys": "091803G1_phys.json"},
    "072051G1": {"name": "原斗", "region": "P", "rock": "072051G1_rockrecord.json", "phys": "072051G1_phys_null.json"},
    "DHW-08": {"name": "東源", "region": "MT", "rock": "DHW-08_rockrecord.json", "phys": "DHW-08_phys.json"},
}

SOURCES = [
    {"original": "1794902460867.json", "raw": "091803G1_rockrecord.json", "sha256": "C9186218F4C9284AD906655A5141CD759398C04CD735548D75EA3DF42757B10C", "kind": "RockRecord", "station": "091803G1", "url": "https://hydro.geologycloud.tw/data/P/RockRecord?station=091803G1", "included": True, "note": "明德岩性；POST响应"},
    {"original": "1794902476552.json", "raw": "091803G1_phys.json", "sha256": "FC5C6FBF1FB1B8944122C3AED1C0445D172F30EF678703D4798AEEACB8A1B629", "kind": "phys", "station": "091803G1", "url": "https://hydro.geologycloud.tw/data/P/phys?station=091803G1", "included": True, "note": "明德井测；POST响应"},
    {"original": "1794902994990.json", "raw": "", "sha256": "FC5C6FBF1FB1B8944122C3AED1C0445D172F30EF678703D4798AEEACB8A1B629", "kind": "phys", "station": "091803G1", "url": "https://hydro.geologycloud.tw/data/P/phys?station=091803G1", "included": False, "note": "与1794902476552.json完全重复，未复制"},
    {"original": "1794903196664.json", "raw": "072051G1_rockrecord.json", "sha256": "AB56E912B45696899A088457F5EE7D619BD8205A2F9E72C230CE5E9276D5EFBB", "kind": "RockRecord", "station": "072051G1", "url": "https://hydro.geologycloud.tw/data/P/RockRecord?station=072051G1", "included": True, "note": "原斗岩性；POST响应"},
    {"original": "1794903344229.json", "raw": "072051G1_phys_null.json", "sha256": "A9E86B39732F657FB42ED3A74C051E064F435C4BB048D8EDA874D7687BCC1980", "kind": "phys", "station": "072051G1", "url": "https://hydro.geologycloud.tw/data/P/phys?station=072051G1", "included": True, "note": "原斗井测Data:null；不是零值"},
    {"original": "response_1794903412630.json", "raw": "DHW-08_rockrecord.json", "sha256": "7AF33FA3F3E9613769361C3B1F4D7824B23EA694A9A8D1BE26405817C5873209", "kind": "RockRecord", "station": "DHW-08", "url": "https://hydro.geologycloud.tw/data/MT/RockRecord?station=DHW-08", "included": True, "note": "东源岩性；山区数字代码"},
    {"original": "response_1794903497864.json", "raw": "DHW-08_phys.json", "sha256": "C165940063ECA21499F8A225B8EFE70B1EF459AB86E3FA85D43A6D8F7250F8EF", "kind": "phys", "station": "DHW-08", "url": "https://hydro.geologycloud.tw/data/MT/phys?station=DHW-08", "included": True, "note": "东源井测；POST响应"},
    {"original": "response_1794903732426.json", "raw": "P_HydroStation.geojson", "sha256": "98B8A389FA6837AE42AB4251B0BDDE46D770F44166935627505C8AF1CCDED01B", "kind": "HydroStation", "station": "P_ALL", "url": "https://hydro.geologycloud.tw/data/P/HydroStation?f=geojson", "included": True, "note": "平原423孔元数据"},
    {"original": "response_1794903865523.json", "raw": "MT_HydroStation.geojson", "sha256": "9674B22ECA5FE093DC86EF6E1485446274A2706A79889037064B5E02395815FC", "kind": "HydroStation", "station": "MT_ALL", "url": "https://hydro.geologycloud.tw/data/MT/HydroStation?f=geojson", "included": True, "note": "山区131孔元数据"},
]

VERIFIED_INTERVAL_LABELS = {
    ("091803G1", 0.1, 3.7, "vfS"): "極細砂",
    ("091803G1", 3.7, 4.7, "ZfS"): "細砂含粉砂",
    ("091803G1", 11.1, 12.6, "Z"): "粉砂",
    ("091803G1", 12.8, 15.0, "Z"): "粉砂",
    ("091803G1", 15.8, 15.9, "Z"): "粉砂",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_raw() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in SOURCES:
        source = DOWNLOADS / item["original"]
        row = dict(item)
        row["access_date"] = ACCESS_DATE
        row["original_download_path"] = str(source)
        if not item["included"]:
            row["copy_verified"] = bool(source.exists() and sha256(source) == item["sha256"])
            manifest.append(row)
            continue
        target = RAW / item["raw"]
        if not target.exists():
            if not source.exists():
                raise FileNotFoundError(f"缺少原始下载文件: {source}")
            if sha256(source) != item["sha256"]:
                raise ValueError(f"原始下载文件哈希不匹配: {source}")
            shutil.copy2(source, target)
        if sha256(target) != item["sha256"]:
            raise ValueError(f"项目内原始文件哈希不匹配: {target}")
        row["copy_verified"] = True
        row["raw_path"] = str(target)
        manifest.append(row)
    return pd.DataFrame(manifest)


def metadata_tables() -> tuple[pd.DataFrame, dict]:
    rows = []
    features_by_station = {}
    for region, raw_name, url in [
        ("P", "P_HydroStation.geojson", "https://hydro.geologycloud.tw/data/P/HydroStation?f=geojson"),
        ("MT", "MT_HydroStation.geojson", "https://hydro.geologycloud.tw/data/MT/HydroStation?f=geojson"),
    ]:
        geo = load_json(RAW / raw_name)
        for feature in geo["features"]:
            props = feature["properties"]
            station = props["STATION_NMBR"]
            if station not in TARGETS or TARGETS[station]["region"] != region:
                continue
            row = dict(props)
            row["source_region"] = region
            row["longitude_wgs84"] = feature["geometry"]["coordinates"][0]
            row["latitude_wgs84"] = feature["geometry"]["coordinates"][1]
            row["hydrostation_url"] = url
            row["access_date"] = ACCESS_DATE
            rows.append(row)
            features_by_station[station] = feature
    if set(features_by_station) != set(TARGETS):
        raise ValueError(f"目标井元数据不完整: {sorted(features_by_station)}")
    metadata = pd.DataFrame(rows)
    first = ["STATION_NMBR", "OFFICIAL_NAME", "STATION_NAME", "source_region", "x97", "y97", "longitude_wgs84", "latitude_wgs84"]
    metadata = metadata[first + [c for c in metadata.columns if c not in first]]
    metadata["existing_888_target_overlap"] = True
    metadata["new_information_scope"] = "depth-resolved lithology and geophysical logs; not a new independent well-location validation set"
    return metadata, features_by_station


def lithology_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for station, config in TARGETS.items():
        payload = load_json(RAW / config["rock"])
        source = SOURCES[next(i for i, x in enumerate(SOURCES) if x["raw"] == config["rock"])]
        intervals = payload.get("data", {}).get("Data") or []
        for i, item in enumerate(intervals, start=1):
            top = item["TOP_DEPTH"]
            bottom = item["BOTTOM_DEPTH"]
            code = str(item["TYPE_OF_ROCK"])
            key = (station, float(top), float(bottom), code)
            label = VERIFIED_INTERVAL_LABELS.get(key, "")
            if code == "L" and config["region"] == "P":
                label = "資料空缺"
                evidence = "明德岩心影像截图中相同代码L对应資料空缺"
            elif label:
                evidence = "明德岩心影像截图的相同井号、深度和代码直接对应"
            else:
                evidence = ""
            if code == "L" and config["region"] == "P":
                label_status = "missing_label_from_verified_plain_borehole_example"
            elif label:
                label_status = "directly_verified_interval"
            else:
                label_status = "not_translated_pending_official_dictionary"
            records.append({
                "station_id": station,
                "well_name": config["name"],
                "source_region": config["region"],
                "interval_index": i,
                "top_depth_m": top,
                "bottom_depth_m": bottom,
                "thickness_m": round(bottom - top, 6),
                "rock_code": code,
                "verified_label_for_this_interval": label,
                "label_evidence": evidence,
                "label_status": label_status,
                "explicit_missing_code": code == "L" and config["region"] == "P",
                "source_file": config["rock"],
                "request_url": source["url"],
            })
    lith = pd.DataFrame(records)
    grouped = []
    for (region, code), part in lith.groupby(["source_region", "rock_code"], sort=True):
        examples = part[part["verified_label_for_this_interval"] != ""]
        example_label = examples.iloc[0]["verified_label_for_this_interval"] if len(examples) else ""
        example_interval = ""
        if len(examples):
            r = examples.iloc[0]
            example_interval = f"{r.station_id} {r.top_depth_m:g}–{r.bottom_depth_m:g} m"
        grouped.append({
            "source_region": region,
            "rock_code": code,
            "interval_count": len(part),
            "total_thickness_m": round(float(part["thickness_m"].sum()), 6),
            "observed_stations": ";".join(sorted(part["station_id"].unique())),
            "minimum_top_depth_m": part["top_depth_m"].min(),
            "maximum_bottom_depth_m": part["bottom_depth_m"].max(),
            "verified_example_label": example_label,
            "verified_example_interval": example_interval,
            "dictionary_status": "only_example_label_verified_not_full_dictionary" if example_label else "official_dictionary_not_yet_obtained",
        })
    return lith, pd.DataFrame(grouped)


def phys_table() -> pd.DataFrame:
    records = []
    fields = ["SP", "N16", "N64", "GAMMA"]
    for station, config in TARGETS.items():
        payload = load_json(RAW / config["phys"])
        source = SOURCES[next(i for i, x in enumerate(SOURCES) if x["raw"] == config["phys"])]
        values = payload.get("data", {}).get("Data") or []
        for item in values:
            missing = [f for f in fields if item.get(f) is None]
            records.append({
                "station_id": station,
                "well_name": config["name"],
                "source_region": config["region"],
                "depth_m": item["d"],
                "SP": item.get("SP"),
                "N16": item.get("N16"),
                "N64": item.get("N64"),
                "GAMMA": item.get("GAMMA"),
                "missing_fields": ";".join(missing),
                "all_four_missing": len(missing) == 4,
                "source_file": config["phys"],
                "request_url": source["url"],
            })
    return pd.DataFrame(records)


def coverage_table(lith: pd.DataFrame, phys: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, config in TARGETS.items():
        meta = metadata[metadata["STATION_NMBR"] == station].iloc[0]
        rock = lith[lith["station_id"] == station].sort_values("top_depth_m")
        log = phys[phys["station_id"] == station].sort_values("depth_m") if len(phys) else phys
        internal_gap = 0.0
        for i in range(1, len(rock)):
            internal_gap += max(0.0, float(rock.iloc[i]["top_depth_m"]) - float(rock.iloc[i - 1]["bottom_depth_m"]))
        row = {
            "station_id": station,
            "well_name": config["name"],
            "source_region": config["region"],
            "well_depth_m": meta["Well_Depth"],
            "elevation_m": meta["CUR_ELEVATION"],
            "rock_interval_count": len(rock),
            "rock_min_top_depth_m": rock["top_depth_m"].min(),
            "rock_max_bottom_depth_m": rock["bottom_depth_m"].max(),
            "rock_top_unrecorded_gap_m": rock["top_depth_m"].min(),
            "rock_internal_unrecorded_gap_m": round(internal_gap, 6),
            "explicit_L_interval_count": int(rock["explicit_missing_code"].sum()),
            "explicit_L_thickness_m": round(float(rock.loc[rock["explicit_missing_code"], "thickness_m"].sum()), 6),
            "phys_row_count": len(log),
        }
        if len(log):
            depths = log["depth_m"].astype(float).to_numpy()
            diffs = np.diff(np.unique(depths))
            row["phys_raw_min_depth_m"] = float(depths.min())
            row["phys_raw_max_depth_m"] = float(depths.max())
            row["phys_median_depth_step_m"] = float(np.median(diffs)) if len(diffs) else None
            measured = log[["SP", "N16", "N64", "GAMMA"]].notna().any(axis=1)
            row["phys_last_depth_with_any_measurement_m"] = float(log.loc[measured, "depth_m"].max()) if measured.any() else None
            row["phys_all_four_missing_rows"] = int(log["all_four_missing"].sum())
            for field in ["SP", "N16", "N64", "GAMMA"]:
                row[f"{field}_nonmissing_count"] = int(log[field].notna().sum())
                row[f"{field}_missing_count"] = int(log[field].isna().sum())
        else:
            for key in ["phys_raw_min_depth_m", "phys_raw_max_depth_m", "phys_median_depth_step_m", "phys_last_depth_with_any_measurement_m", "phys_all_four_missing_rows"] + [f"{f}_{suffix}" for f in ["SP", "N16", "N64", "GAMMA"] for suffix in ["nonmissing_count", "missing_count"]]:
                row[key] = None
        row["existing_888_target_overlap"] = True
        row["evidence_level"] = "borehole lithology/geophysical proxy; not CI, CBR, plate-bearing or vehicle-pass ground truth"
        rows.append(row)
    return pd.DataFrame(rows)


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def overlap_table(metadata: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(EXISTING_CSV, dtype=str, keep_default_na=False)
    existing.insert(0, "existing_csv_row", np.arange(len(existing)) + 2)
    selected = existing[existing["名称"].isin(["明德", "原斗", "東源"]) | existing["名称"].str.match(r"^明德\([1-4]\)$")].copy()
    target_name = {"明德": "091803G1", "原斗": "072051G1", "東源": "DHW-08"}
    stations, bases, distances = [], [], []
    for _, row in selected.iterrows():
        name = row["名称"]
        station = target_name.get(name, "091803G1")
        meta = metadata[metadata["STATION_NMBR"] == station].iloc[0]
        distance = haversine_m(float(row["经度"]), float(row["纬度"]), float(meta["longitude_wgs84"]), float(meta["latitude_wgs84"]))
        stations.append(station)
        distances.append(distance)
        if name in target_name:
            bases.append("same borehole: exact name plus matching depth, elevation and coordinates")
        else:
            bases.append("separate monitoring-well record near 明德 borehole; do not merge as identical well ID")
    selected.insert(1, "matched_api_station_id", stations)
    selected.insert(2, "distance_to_api_well_m", distances)
    selected.insert(3, "match_basis", bases)
    selected["independence_note"] = "already present in the 888-record compilation; new API attributes are not independent vehicle-pass validation"
    return selected


def dictionary_table() -> pd.DataFrame:
    rows = [
        ["lithology_intervals", "TOP_DEPTH/BOTTOM_DEPTH", "m", "API逐层上下限深度；原始值保留"],
        ["lithology_intervals", "TYPE_OF_ROCK", "source code", "平原字母代码与山区数字代码分开保留；未获得完整官方代码表"],
        ["geophysical_logs", "d / depth_m", "m", "API深度字段d；东源原始记录延伸至100.04 m，未按井深截断"],
        ["geophysical_logs", "SP", "mV (website display)", "单位来自地图柱状图标题，不是由数值反推"],
        ["geophysical_logs", "N16", "ohm-m (website display)", "16英寸电阻率；缺失保留为空"],
        ["geophysical_logs", "N64", "ohm-m (website display)", "64英寸电阻率；缺失保留为空，不插值"],
        ["geophysical_logs", "GAMMA", "cps (website display)", "Gamma；缺失保留为空"],
        ["all", "null / blank", "missing", "缺失不是0；原斗phys Data:null保留原始JSON，不生成曲线行"],
        ["existing_888_overlap", "nan string in legacy fields", "legacy missing token", "从现有CSV原样保留，表示源字段缺失；不是实测数值或0"],
        ["existing_888_overlap", "existing_csv_row", "1-based including header", "point_validation_corrected.csv中的实际行号"],
        ["scope", "hydraulic conductivity/transmissivity/pumping rate/specific drawdown", "source units", "旧汇编四个字段的正式单位仍未由本API解决，不擅改"],
        ["scope", "CI/CBR/plate bearing/VCI/Go-No-Go", "not available", "本批次没有现场机械真值或车辆通行试验"],
        ["reuse", "license/citation", "unresolved", "操作手册未检出授权或引用条款；公开再发布前需另行核对网站条款"],
    ]
    return pd.DataFrame(rows, columns=["table_or_scope", "field_or_item", "unit_or_status", "rule_or_note"])


def write_csv(df: pd.DataFrame, name: str):
    df.to_csv(BASE / name, index=False, encoding="utf-8-sig", na_rep="")


def write_workbook(sheets: dict[str, pd.DataFrame], path: Path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            if len(df.columns):
                ws.auto_filter.ref = f"A1:{get_column_letter(len(df.columns))}{len(df) + 1}"
            if len(df) > 30:
                ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="DCE6F1")
                cell.font = Font(name="微软雅黑", bold=True, color="1F1F1F")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for idx, col in enumerate(df.columns, start=1):
                sample = [str(col)] + [str(v) for v in df[col].dropna().head(200)]
                width = min(50, max(10, max((len(v) for v in sample), default=10) + 2))
                ws.column_dimensions[get_column_letter(idx)].width = width
                if any(token in str(col) for token in ["station_id", "STATION_NMBR", "OFFICIAL_NAME", "rock_code"]):
                    for cell in ws[get_column_letter(idx)][1:]:
                        cell.number_format = "@"
                if any(token in str(col).lower() for token in ["note", "evidence", "url", "scope", "basis"]):
                    for cell in ws[get_column_letter(idx)][1:]:
                        cell.alignment = Alignment(vertical="top", wrap_text=True)


def write_readme(coverage: pd.DataFrame, metadata: pd.DataFrame):
    lines = [
        "# 臺灣水文地質資訊系統：三口井API小样",
        "",
        f"获取与整理日期：{ACCESS_DATE}",
        "",
        "## 范围",
        "",
        "本批次仅整理明德、原斗、东源的井位元数据、逐层岩性和井测曲线，并保存平原423孔与山区131孔的井位GeoJSON。未批量下载其他井的岩性/井测，未下载岩心影像、碳十四、水位或水力试验记录。",
        "",
        "三口目标井均已在现有888条汇编中。本次新增价值是随深度变化的属性，不是新井位，也不是独立车辆通行验证集。",
        "",
        "## 覆盖与缺失",
        "",
        "|井|岩性层数|岩性深度范围(m)|顶部未记录(m)|内部未记录累计(m)|井测行数|井测原始深度范围(m)|最后有任一量测深度(m)|备注|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in coverage.iterrows():
        def f(v):
            return "" if pd.isna(v) else f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)
        note = "phys Data:null" if r["phys_row_count"] == 0 else f"N64缺失{int(r['N64_missing_count'])}行"
        lines.append(f"|{r['well_name']} {r['station_id']}|{r['rock_interval_count']}|{f(r['rock_min_top_depth_m'])}–{f(r['rock_max_bottom_depth_m'])}|{f(r['rock_top_unrecorded_gap_m'])}|{f(r['rock_internal_unrecorded_gap_m'])}|{r['phys_row_count']}|{f(r.get('phys_raw_min_depth_m'))}–{f(r.get('phys_raw_max_depth_m'))}|{f(r.get('phys_last_depth_with_any_measurement_m'))}|{note}|")
    lines += [
        "",
        "- 原斗岩性从4.2 m开始，不能据此描述0–4.2 m的实测岩性；其井测响应为`Data:null`。",
        "- 东源井测原始记录到100.04 m，保留原值，不按100 m井深擅自截断；末段全空行与真实量测分开统计。",
        "- 平原`L`按截图直接对应记为“资料空缺”；其他岩性代码只有在相同井号、深度和代码直接对应时才填写中文标签。",
        "- 山区数字岩性代码与平原字母代码不混用，完整官方代码表尚未取得。",
        "- `existing_888_overlap.csv`中旧字段的`nan`字符串是源汇编的缺失标记，不是实测数值或0。",
        "",
        "## 文件",
        "",
        "- `raw/`：8份去重后的原始JSON/GeoJSON；下载清单另记录1份完全重复文件，未复制。",
        "- `well_metadata.csv`：三口井元数据。",
        "- `lithology_intervals.csv`：逐层岩性与缺失代码。",
        "- `geophysical_logs.csv`：SP、N16、N64、Gamma逐深度记录，保留空值。",
        "- `rock_code_observations.csv`：代码出现次数、厚度和有限标签证据，不是完整字典。",
        "- `coverage_summary.csv`：深度覆盖、采样间隔和字段缺失统计。",
        "- `existing_888_overlap.csv`：三口井及明德4条监测井的现有记录与行号。",
        "- `download_manifest.csv`：原始文件名、API地址和SHA-256。",
        "- `水文地质API小样_三口井.xlsx`：上述核心表的多子表工作簿。",
        "- `analysis_summary.json`：机器可读摘要。",
        "- `build_hydro_sample.py`：确定性整理脚本。",
        "",
        "## 证据边界",
        "",
        "这些数据可用于近地表岩性、沉积环境和地球物理代理分析，不能换算或宣称为CI、CBR、平板承载、VCI或车辆Go/No-Go真值。单井与30 m像元存在尺度差异，同位置多井/多记录也不能当作独立样本。",
        "",
        "旧汇编中hydraulic conductivity、transmissivity、pumping rate、specific drawdown的正式单位仍未由本API解决，继续保留`source units`。",
        "",
        "## 来源与再使用",
        "",
        f"- 地图：{MAP_URL}",
        f"- API：{API_DOC}",
        f"- 操作手册：{HELP_URL}",
        "",
        "操作手册未检出明确的授权或引用条款。本次仅完成查询与本地整理，未确认公开再发布许可；若上传GitHub/Zenodo或公开附件，需先核对网站最新条款并保留来源。",
    ]
    (BASE / "README_水文地质API小样.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    manifest = prepare_raw()
    metadata, _ = metadata_tables()
    lith, codes = lithology_tables()
    phys = phys_table()
    coverage = coverage_table(lith, phys, metadata)
    overlap = overlap_table(metadata)
    dictionary = dictionary_table()

    outputs = {
        "download_manifest.csv": manifest,
        "well_metadata.csv": metadata,
        "lithology_intervals.csv": lith,
        "geophysical_logs.csv": phys,
        "rock_code_observations.csv": codes,
        "coverage_summary.csv": coverage,
        "existing_888_overlap.csv": overlap,
        "data_dictionary.csv": dictionary,
    }
    for name, df in outputs.items():
        write_csv(df, name)

    sheets = {
        "覆盖摘要": coverage,
        "井位元数据": metadata,
        "岩性分层": lith,
        "井测曲线": phys,
        "岩性代码观察": codes,
        "现有888条重叠": overlap,
        "字段与限制": dictionary,
        "下载清单": manifest,
    }
    workbook = BASE / "水文地质API小样_三口井.xlsx"
    write_workbook(sheets, workbook)
    write_readme(coverage, metadata)

    summary = {
        "access_date": ACCESS_DATE,
        "target_stations": list(TARGETS),
        "raw_unique_files": int(manifest["included"].sum()),
        "duplicate_files_excluded": int((~manifest["included"]).sum()),
        "lithology_rows": len(lith),
        "geophysical_rows": len(phys),
        "overlap_rows": len(overlap),
        "coverage": json.loads(coverage.to_json(orient="records", force_ascii=False)),
        "workbook_sha256": sha256(workbook),
        "limitations": [
            "All three target boreholes already exist in the 888-record compilation.",
            "No CI, CBR, plate-bearing, VCI or vehicle Go/No-Go ground truth was obtained.",
            "The complete official lithology-code dictionary was not obtained.",
            "The website reuse/redistribution license was not confirmed.",
            "Formal units of the four legacy hydrogeological fields remain unresolved.",
        ],
    }
    (BASE / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "success", "base": str(BASE), "lithology_rows": len(lith), "geophysical_rows": len(phys), "overlap_rows": len(overlap), "workbook": str(workbook)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
