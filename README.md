# 台西平原相对交通适宜性筛查：U-net 与 SegFormer 代码说明

本项目使用 **U-net（CNN）** 与 **SegFormer（Transformer）** 对13通道遥感特征进行像素级三分类，用于相对交通适宜性筛查、参考标签比较和模型机制诊断。

> 输出不是绝对承载力、设计荷载或指定车辆的 Go/No-Go 结论。AHP/机理标签属于参考或代理标签，不能替代现场 cone index、CBR、平板承载或车辆通行试验。

## 1. 当前英文稿采用的实验协议

### 全岛 AHP 参考标签实验

| 项目 | 当前口径 |
|---|---|
| 输入 | 13通道，128×128完全有效瓦片 |
| 有效瓦片 | 2,045 |
| 训练/验证/测试 | 1,232 / 414 / 399 |
| 模型 | U-net、SegFormer B0、SegFormer B2 |
| 随机种子 | 42、123、2024；每个种子分别训练和评估 |
| 优化 | Adam，初始学习率1e-4，未加权多类交叉熵 |
| 训练 | 最多30 epochs；按验证集 macro-F1 保存最佳状态 |
| 全岛 batch size | 16 |
| 数据增强 | 水平/垂直翻转及90°整数倍旋转；不包含其他已删除增强 |
| 划分性质 | 确定性哈希分配，不是空间阻塞或空间独立验证 |

### 共同域参考标签比较

| 项目 | 当前口径 |
|---|---|
| 瓦片 | 771；546训练、225验证 |
| 测试集 | 无独立测试集；结果为共同域验证比较 |
| 模型 | U-net、SegFormer、RF、RBF-SVM |
| batch size | U-net 16；SegFormer 32 |
| 结果解释 | 验证集参与最佳状态选择，不冒充独立测试精度 |

### 网络与校正RCI

- U-net：9个 DoubleConv，共18个3×3卷积，加1个1×1输出卷积，总计19层。
- SegFormer：首段7×7/stride 4，后三段3×3/stride 2；输入13通道，直接输出3类。
- 校正机理链：`MC(%) = W33 × 1000 / BD`，`RCI = exp(c1 − c2 ln MC)`；不额外添加SOC或remolding乘子。
- OpenLandMap使用表层 `b0`；BD按源产品换算为 kg/m³。W33仅为水分状态代理，不是同期实测含水率。
- 四个深层水文字段仍保留 `source units`，不能根据口头量纲线索统一补写正式单位。

## 2. 代码目录与版本边界

```text
trafficability_dl/
├── config.py        # 早期通用配置；其中部分默认值不是最终论文协议
├── data_utils.py    # GeoTIFF读取、切片、Dataset及通用数据处理
├── unet.py          # 19层U-net实现
├── segformer.py     # SegFormer MiT编码器与MLP解码器
├── metrics.py       # 混淆矩阵、OA、Kappa、PA/UA、F1、AUC
├── train.py         # 通用训练入口，不等于最终全岛九次实验驱动
├── real_train.py    # 早期真实数据流程与调试入口
├── smoke_test.py    # 合成数据自检，不生成论文结果
└── requirements.txt
```

**重要限制：** 当前目录尚未定位到一个完整覆盖“全岛三模型 × 三种子 × 1,232/414/399划分”的最终训练入口，也未集中包含全部最终划分清单、权重和 canonical 输出。因此，不能把本目录描述为已经一键复现论文全部实验。`config.py`、`train.py` 或 `real_train.py` 中若仍出现256×256、batch 20、50 epochs、空间7:3或旧阈值，均属于早期通用/调试口径，不能覆盖上表中的最终协议。

在公开声称完全可复现前，应补齐：

1. 最终三模型三种子的统一训练驱动与固定配置；
2. 1,232/414/399及546/225瓦片索引或可再生分配文件；
3. 最终权重、canonical混淆矩阵和评价输出的一一对应关系；
4. Figure 6–8及配对10,000次瓦片bootstrap的生成入口；
5. 环境版本、依赖锁文件和运行说明。

## 3. 输入数据

特征GeoTIFF应为13波段、30 m、float32，研究区外为nodata。波段顺序为：

`NLI, NBD, NCLAY, NSAL, NAI, NSI, NCUR, NDW, NWDI, NSAR, NLST, NVFI, NLULC`

标签取值为0/1/2，对应相对低/中/高等级。连续AHP分数需按当前稿件的Jenks流程重新计算，不应沿用早期固定阈值。共同域实验复用companion研究的13通道输入和连续BCI，但本文重新计算Jenks并训练模型，不复用其图表。

大型派生标签栅格不在轻量Zenodo归档中；第三方数据仍受原始来源的访问和再分发条件约束。

## 4. 通用自检与训练

安装依赖后可运行合成数据自检：

```bash
python smoke_test.py
```

通用训练入口为：

```bash
python train.py --model unet
python train.py --model segformer
```

这些命令只说明通用代码的使用方式。运行前必须显式核对数据路径、通道顺序、标签、补丁大小、划分文件、batch size、epoch和seed；不能用早期默认配置直接复现并声称得到论文最终结果。

## 5. 结果解释与评价边界

- OA、Kappa、PA/UA、F1和AUC首先衡量对参考标签的复现程度，不是独立机械真值验证。
- 共同域验证集参与模型选择，不能写成held-out test。
- 空间置换仅为单种子、单分配机制诊断，不能推广为所有CNN或Transformer的普遍规律。
- 水文记录用于敏感性分析；共享坐标不是唯一井号，记录级p值未建模同位置聚类。
- 绝对车辆通行判断仍需现场CI/CBR/平板承载、同期水分、车辆参数和仪器化通行试验。

## 6. 公开链接

- GitHub：https://github.com/zishu206/trafficability-taixi
- Zenodo概念DOI：https://doi.org/10.5281/zenodo.24189358
- Zenodo v1.0.0版本DOI：https://doi.org/10.5281/zenodo.24189361

本地说明更新不等于公开GitHub或Zenodo已经同步；公开状态应以实际提交和release记录为准。
