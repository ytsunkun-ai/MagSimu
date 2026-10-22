# 数据目录

## `raw/`

- `example_bh_placeholder.csv`：`H_A_m`, `B_T`, `branch`, `frequency_hz`, `source`
- `example_loss_placeholder.csv`：`frequency_hz`, `B_peak_T`, `loss_W_kg`, `loss_total_W_m3`, `loss_hysteresis_W_m3`, `loss_eddy_W_m3`, `loss_excess_W_m3`, `temperature_K`, `source`
- `example_breakdown_placeholder.csv`：`breakdown_voltage_V`, `thickness_m`, `breakdown_field_V_m`, `source`

这些文件由 `make_placeholders.py` 生成，全部是 **synthetic placeholder**，仅用于验证代码路径。
`source` 列中的标记会被校准模块识别，并在结果中给出 `data_quality=synthetic_placeholder` 警告。

## `processed/`

预留给经过清洗、带单位和溯源信息的数据。不要在这里放未注明来源的数值。

## 替换实测数据时

1. 保留列名或使用 `calibration.py` 支持的别名。
2. 在 `source` 列写明实验批次/文件/日期，避免把 placeholder 与实测数据混合。
3. 更新 `configs/experiment.yaml` 中的路径。
4. 运行 `softmag run-all`，检查 `calibration/summary.json` 的 `data_quality`、R² 和 `at_bound`。
