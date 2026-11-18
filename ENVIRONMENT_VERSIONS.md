# Environment versions

## Anaconda Python used for CPU model inference

- Python: 3.12.7 (Anaconda)
- torch: 2.12.0
- numpy: 1.26.4
- scikit-learn: 1.5.1
- tqdm: 4.66.5
- pandas: 2.2.2
- openpyxl: 3.1.5
- rasterio: not available in this environment

## Default Python used for raster/data-processing checks

- Python: 3.13.18
- numpy: 2.5.4
- rasterio: 1.5.2
- scikit-learn: 1.9.1
- pandas: 3.1.1
- openpyxl: 3.1.5
- torch: not available in this environment
- tqdm: not available in this environment

`requirements.txt` 是最低依赖说明，不是完整锁文件。由于不同步骤使用了不同本机环境，本轻量归档不能描述为已经提供一键复现全部实验的统一环境。
