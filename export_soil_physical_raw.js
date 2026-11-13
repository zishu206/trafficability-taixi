/****
 * 台西平原 机理标签 原始量纲土壤参数导出（GEE JavaScript）
 * 产品：OpenLandMap（原生 250 m），取 0 cm 表层 band（b0），保留原始量纲、不做归一化
 * 输出：EPSG:4326、30 m，用 crsTransform 与现有 TaiXi_13indicators_30m.tif 严格配准
 * 分块：2 列 × 3 行 = 6 个导出任务，运行后在右侧 Tasks 面板逐个 Run
 * 若某 Image ID 或 band 名报错：在 Code Editor 输入前缀用自动补全核对，
 * 或先运行 print(ee.Image('OpenLandMap/SOL/...').bandNames()) 查看实际波段名。
 ****/

// ===== 1. 研究区范围（与现有 13 通道 tif 同 bounds）=====
var X_MIN = 119.30767833518078;
var Y_MIN = 21.90155544753282;
var X_MAX = 121.00010433046197;
var Y_MAX = 24.95978000078932;
var roi = ee.Geometry.Rectangle([X_MIN, Y_MIN, X_MAX, Y_MAX], null, false);

// 与现有 tif 完全一致的像元变换，保证零配准偏移
var TRANSFORM = [
  0.00026949458523585647, 0, X_MIN,
  0, -0.00026949458523585647, Y_MAX
];

// ===== 2. 加载 OpenLandMap 各产品，取表层 b0，单位换算并命名 =====
function surf(id) { return ee.Image(id).select('b0'); }

var SAND = surf('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02').rename('SAND');        // 砂粒 %
var CLAY = surf('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02').rename('CLAY');        // 黏粒 %
// GEE 无独立粉粒产品，按 USDA 三组分质量分数和为 100 由砂、黏推算（海域随砂/黏保持 null）
var SILT = ee.Image(100).subtract(SAND).subtract(CLAY).rename('SILT');                          // 粉粒 %（=100-SAND-CLAY）
var BD   = surf('OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02').multiply(10).rename('BD');  // 容重：10×kg/m3 -> kg/m3
var SOC  = surf('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02').multiply(5).rename('SOC');     // 有机碳：×5 g/kg -> g/kg
var W33  = surf('OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v02').rename('W33');       // 田持(33kPa)含水率 体积%
var TEX  = surf('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02').rename('TEX');             // USDA 质地类别码（可选）

// ===== 3. 合成多波段并裁剪到研究区（默认最近邻，忠实保留 250m 父像元值，不虚构信息）=====
// 统一转 Float32：质地码为 Byte、其余为 Int16，类型不一致会导致导出失败（Error code 3）
var soil = ee.Image.cat([SAND, SILT, CLAY, BD, SOC, W33, TEX]).toFloat().clip(roi);

// ===== 4. 分块网格（如报内存超限，把 NX、NY 调大，块更小）=====
var NX = 2, NY = 3;
var dx = (X_MAX - X_MIN) / NX;
var dy = (Y_MAX - Y_MIN) / NY;
var cells = [];
for (var ix = 0; ix < NX; ix++) {
  for (var iy = 0; iy < NY; iy++) {
    var x0 = X_MIN + ix * dx, x1 = X_MIN + (ix + 1) * dx;
    var y0 = Y_MIN + iy * dy, y1 = Y_MIN + (iy + 1) * dy;
    cells.push({ name: 'r' + ix + '_' + iy, geom: ee.Geometry.Rectangle([x0, y0, x1, y1], null, false) });
  }
}

// ===== 5. 逐块导出到 Google Drive（Tasks 面板逐个 Run）=====
cells.forEach(function (c) {
  Export.image.toDrive({
    image: soil,
    description: 'soil_physical_raw_' + c.name,
    folder: 'GEE_taixi',
    region: c.geom,
    crs: 'EPSG:4326',
    crsTransform: TRANSFORM,
    maxPixels: 1e13
  });
});

// 可选：在地图上预览研究区与第一波段
Map.centerObject(roi, 8);
Map.addLayer(roi, { color: 'red' }, '研究区');
Map.addLayer(SAND, { min: 0, max: 100, palette: ['ffffcc', '41b6c4', '253494'] }, '砂粒% 预览');
