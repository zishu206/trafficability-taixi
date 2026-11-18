# Data provenance

## 1. Sources

- Hydrogeology map: https://hydro.geologycloud.tw/map
- API documentation: https://hydro.geologycloud.tw/swagger/api-docs/api
- Help page: https://hydro.geologycloud.tw/help/help.html
- data.gov dataset 16799, Hydrogeological drilling database: https://data.gov.tw/dataset/16799
- data.gov dataset 61131, plain borehole core records: https://data.gov.tw/dataset/61131
- data.gov dataset 61132, plain borehole well logs: https://data.gov.tw/dataset/61132
- Government Open Licence page: https://data.gov.tw/license?lang=zh-tw

主要访问和整理日期为2026-11-17；本v1.1.0上传暂存包整理日期为2026-11-18。

## 2. Scope of the added evidence

本次新增的是明德、原斗和东源三口井随深度变化的岩性、粒径代码和地球物理测井摘要。三口井的井位本身已经存在于项目888条汇编中，因此不能把它们包装成新发现的独立验证井位。

`supporting_validation/` 只包含小体积派生表和说明，不包含原始API JSON、大型栅格、模型权重或岩心照片。

## 3. Missing-value rules

- 平原代码 `L` 表示“资料空缺”，不是实测岩性，也不是零值。
- SP、16英寸电阻率、64英寸电阻率和Gamma中的缺失值保留为空，不进行插值或补造。
- 原斗井测接口返回 `Data:null`，该状态表示没有可用井测记录，不能解释为零。
- 61132全量井录在质量服务下载时遇到锁等待/500错误；本包不包含未取得的全量文件。
- 旧汇编中hydraulic conductivity、transmissivity、pumping rate和specific drawdown仍保留source units。关于透水系数可能为m/day的口头信息没有正式元数据支持，未写入论文或本数据包。

## 4. Licence and redistribution boundary

data.gov相关数据集页面载明适用“政府資料開放授權條款-第1版”，计费方式为免费，并要求显名声明。水文地图、Swagger/API页面本身未单列独立再发布条款，因此本包不包含原始API JSON，以避免超出已确认的再发布边界。

## 5. Evidence boundary

这些资料只能作为近地表岩性、沉积环境和地球物理方面的supporting proxy validation。它们不包含CI、CBR、平板承载、车辆荷载、设计荷载或实际车辆pass/fail字段，不能换算或宣称为独立机械验证结果。
