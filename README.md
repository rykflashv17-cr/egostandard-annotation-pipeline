# EgoStandard 四阶段标签修复与时间表管线

按 S1 → S2 → S3 → S4 顺序运行，S3 可选；输出一份 S1 修复标签和逐原视频的完整合格／不合格时间表。不切分或重编码视频。

## 从仓库运行

```bash
git clone https://github.com/rykflashv17-cr/egostandard-annotation-pipeline.git
cd egostandard-annotation-pipeline
python -m pip install -r requirements.lock.txt
python -m unittest discover -s tests -v
python run.py --source /path/to/stage_a_aligned --output /path/to/work/labels_annotations --workers 64
# 跳过 S3：在最后一条命令增加 --skip-s3
# 恢复同一版本的未完成任务：增加 --resume
```

需要 Python ≥3.10；已在 Python 3.12.8、NumPy 2.5.2、SciPy 1.18.1、PyArrow 25.0.1 上验证。只运行功能测试无需真实数据；正式运行需要原数据的标签和元信息。

部署版本已通过 5 项功能测试与 256 个真实 episode 的两模式集成验证：标签对齐、输出哈希、完整时间轴覆盖和 S3 关闭后的 S4 重算均通过。全量导出尚未启动。

仓库中的 `licenses/` 保留 MINT 上游许可与固定版本信息，`PROVENANCE.json` 记录各规则来源和修改。

## 详细规则与部署说明

本程序依次执行 **S1 → S2 → S3 → S4**。`--skip-s3` 改为 **S1 → S2 → S4**，S4 会在 S2 的剩余连续区间上重新计算。没有 S5。每个阶段只检查前面阶段保留的连续区间，窗口不跨越已标记为不合格的区间。不同 episode 并行运行，同一个 episode 按阶段顺序执行。

输入默认是开发机上的原始对齐数据：

```text
/mnt/pfs/datasets/Processed/Egostandard_stageA_260915_delay1_trunc/stage_a_aligned
```

## 当前规则

| 阶段 | 规则 |
| --- | --- |
| S1 | 现有 MINT block + Hampel 轨迹异常检测；有两侧有效端点且异常长度 ≤30 帧时，位置线性插值，旋转 SO(3) SLERP。无法修复的帧标记为不合格；原始值保留并添加无效标志。 |
| S2 | 双手同时位于画面外或 5% 边缘，连续 ≥2 秒。由 EEF 与相机标定几何投影判断。 |
| S3（默认启用） | 0.2 秒（30 FPS 下间隔 6 帧）两端相机平移 **>7.5 cm** 或相对 SO(3) 旋转 **>7.5°**。 |
| S4 | 1 秒窗口中，双手各自相对相机光轴的余弦 P95−P5 ≤0.03，且相机两端平移 **>20 cm** 或旋转 **>20°**；每个分支独立连续命中 ≥1 秒，最后取并集。 |

S3、S4 标记命中窗口的全部证据帧（包括两端），不额外向外扩展或填补正长度间隙。S4 中“连续命中 ≥1 秒”指 30 个连续窗口起点，所以最短完整证据区间为 60 帧（2 秒）。旋转分支沿用现有程序的 `1e-5°` 数值容差。

## 运行

程序已重新部署在 `/mnt/pfs/Data/ryk/egostandard/program/`。开发机使用现有 Python 环境；启动脚本和 `run.py` 默认输出到 `/mnt/pfs/Data/ryk/egostandard/runs/labels_annotations_20260928/`。程序、测试、运行日志与半成品统一保存在 Data/ryk；Processed 用于读取已有数据和发布验证完成的最终数据。

```bash
bash /mnt/pfs/Data/ryk/egostandard/program/run.sh --workers 64

# 跳过 S3；完成后的 S1 标签直接复用，不再写第二份标签。
bash /mnt/pfs/Data/ryk/egostandard/program/run.sh --workers 64 --skip-s3

# 同一版本中断后恢复：
bash /mnt/pfs/Data/ryk/egostandard/program/run.sh --workers 64 --resume
```

也可直接使用 `python run.py --source INPUT --output OUTPUT --workers 64`。`--limit 256` 只跑前 256 个 episode 的测试，必须使用独立测试输出目录。`--batch-size` 默认 32。每个 worker 的 BLAS/OpenMP 线程限制为 1，避免线程数倍增。

## 输出结构

```text
/mnt/pfs/Data/ryk/egostandard/runs/labels_annotations_20260928/
  config.json                      # 源元信息、代码哈希、全部阈值与输出策略
  labels_s1/
    data/chunk-xxx/file-xxx.parquet # 每个原 episode 对应一份 S1 修复标签
    meta/                          # 元信息与原始 episode/source 映射
    manifest.jsonl                 # 原文件身份、新标签哈希、修复细节、S1 无效区间
    COMPLETE.json
  annotations/
    with_s3/                       # S1 → S2 → S3 → S4
      plan.jsonl                   # 每行一个完整源 episode，含视频路径、完整时间轴
      timeline.csv.gz              # 合格与不合格区间共同覆盖完整原视频
      rejected_intervals.csv.gz    # 仅不合格区间
      episodes.csv.gz              # episode、标签与原始两个 MP4 的对应关系
      summary.json                 # 每阶段新增标记的帧数、时长及总量
      COMPLETE.json                # 全部校验成功后才产生
      checkpoints/                 # 批次检查点，用于断点恢复
    without_s3/                    # 仅运行 --skip-s3 后才生成，复用 labels_s1
  LATEST.json
```

### 时间表解释

`timeline.csv.gz` 每行含 `source_episode_index`、`start_frame`、`end_frame_exclusive`、起止秒、时长、`status`（`keep` / `reject`）、`stage`（S1–S4）、`reason`，以及两路原 MP4 中的时间偏移。区间为 **[start, end)**；帧编号是权威依据，秒数是 30 FPS 的六位小数表示。

例如 `[210,240)` 是原视频 7.0–8.0 秒的 30 帧。每个 episode 的表从第 0 帧覆盖至最后一帧，无遗漏、无重叠。同一帧归因于最早命中的阶段；同阶段两分支都命中时记录联合原因，不重复统计时长。`plan.jsonl` 中的时间轴格式为 `[起帧, 结束帧（不含）, 阶段编号, 原因编号]`；阶段 0 为合格。原因编号与文字对应见 `pipeline.py` 的 `REASONS`。

`episodes.csv.gz` 给出原视频路径；`plan.jsonl` 为每个 episode 直接附带两路视频路径和偏移。这些表使用原始 episode 编号。此数据源为 30 FPS；此前预览采样每秒 5 帧不改变原始时间轴帧率。

### 标签解释与训练读取

所有原始标签行、`frame_index`、`episode_index`、`timestamp`、全局 `index`、摄像机、gripper 和原始手部特征保持对应。只写检测命中且可插值的 EEF 位置/旋转；action 随其 delay1 目标同步修复。不合格行保留原始值，这样完整原视频仍与新标签逐帧对应；是否使用由时间表决定。

增加双手标志：`observation.state.eef_valid / eef_repaired / eef_anomaly_reason`、`action.eef_valid / eef_repaired / eef_anomaly_reason`，以及 `observation.state.hand_features_valid`。修复 EEF 不会把原始 keypoints 自动变成修复后的 keypoints；后者需看 `hand_features_valid`。

训练时除读取 `keep` 区间，还应检查所需字段的有效标志，且整个训练样本窗口位于同一合格区间。特别是 delay1 的 action 指向下一帧：无效 state 前一帧的 state 可能合格，但该帧 `action.eef_valid` 为假；不能仅用单帧 `keep` 判断 action 可用。自然 episode 末尾的外部 action 目标保持原值；无法修复时最后一帧归 S1 不合格，不伪造额外视频帧或夹紧 action。

`labels_s1` 是 **只有标签的输出**，不包含视频；训练读取器应按表中的路径访问原视频。原 EEF 统计不能当作修复后统计使用，所以未复制原统计文件；需要训练归一化时，再用有效标签和合格时间表计算。

## 校验与恢复

程序不打开、复制、链接、切分或重编码任何视频，也不覆盖原始 label。写标签后逐列读取校验，记录 SHA256；检查源文件在读写期间身份不变。每个 episode 验证输入行号、完整时间轴和帧数守恒，输出按原始 episode 顺序合并。

`--resume` 校验源文件和已输出标签身份后复用批次检查点；未完成批次重新生成。全局完成后才发布最终时间表与 `COMPLETE.json`。S1 尚未完成时不能切换 S3 模式，须先恢复原模式。源数据、规则或程序版本改变时使用新输出目录，避免混合版本。

功能测试：在 `program/` 中运行 `python -m unittest discover -s tests -v`。小规模集成测试使用独立的 `/mnt/pfs/Data/ryk/egostandard/runs/validation_256_20260928/`，不混入全量产物。

## 工作区布局

```text
/mnt/pfs/Data/ryk/egostandard/
  program/        # 当前独立代码包、规则、依赖版本、测试
  runs/           # 标签、时间表、测试产物及运行记录
  review/         # 复查视频与页面
  archive/        # 需要保留的历史程序和实验材料
  DEPLOYMENT.json # 部署版本和验证状态
```

现有运行环境位于 `/mnt/pfs/Data/wanghao/Data_Clean_StridingAI_Ego/.venv_root_miniforge/`，启动脚本直接复用这个环境。换环境时可设置 `PIPELINE_PYTHON`，换输出目录时可用 `--output` 或 `PIPELINE_OUTPUT`；读取原始数据的位置仍可用 `--source` 指定。
