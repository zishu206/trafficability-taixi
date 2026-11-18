# Scripts provenance and use

本目录保留用于三口井整理、东源模型检查和官方代码字典核查的可追溯脚本。脚本内容未改动，以保留原始执行记录。

部分脚本包含本机绝对路径，因为它们依赖未纳入轻量归档的大型栅格、模型权重、原始下载文件或本地项目目录。公开仓库中运行这些脚本前，需要按本机文件位置配置路径；不能仅凭本目录直接复现全部实验。

| 脚本 | 用途 | 主要外部依赖 |
|---|---|---|
| `three_well/build_hydro_sample.py` | 整理三口井原始API小样、岩性、井测、覆盖统计和工作簿 | 本地原始JSON/GeoJSON、888条汇编 |
| `three_well/build_borehole_proxy_validation.py` | 生成论文使用的三口井代理验证表和证据边界说明 | 水文API小样、代码字典、888条汇编 |
| `three_well/map_wells_to_tiles.py` | 将三口井坐标映射到栅格像元、128瓦片和共同域索引 | 栅格文件、common_domain.npz |
| `dongyuan/infer_dongyuan.py` | 早期AHP/机理模型在东源瓦片的类别与概率检查 | 模型权重、common_domain.npz、项目代码 |
| `dongyuan/infer_dongyuan_corrected.py` | 校正机理口径下东源模型检查并写入JSON结果 | 校正模型权重、校正标签、项目代码 |
| `code_dictionary/find_data_gov_resources.py` | 核查data.gov数据集页面和下载入口 | 网络访问 |
| `code_dictionary/download_and_build_dictionary.py` | 下载官方CSV并构建早期代码字典与来源清单 | data.gov质量服务、pandas |
| `code_dictionary/extract_js_context.py` | 从前端JavaScript上下文中定位岩性代码和关键词 | 本地js_inspect目录 |
| `code_dictionary/build_hydro_code_dictionary_from_js.py` | 从官方前端map.js提取水文字母代码和数字地质代码映射 | 本地map.js |
| `code_dictionary/build_three_well_labels.py` | 基于官方代码字典生成三口井代码标签 | 官方CSV和代码字典 |

本目录不包含原始API JSON、大型栅格或模型权重。
