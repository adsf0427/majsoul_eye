# 数据集可视化与清理（YOLO 检测数据集）

用于**肉眼审查、筛查、清理**自动生成的 YOLO 检测数据集
（`datasets/<game>/yolo/images,labels/` + `datasets/detector/{data.yaml,train.txt,val.txt}`，
38 类，repo-root 相对 POSIX 路径）。两条互补路径：

- **A · FiftyOne** — 浏览器 GUI，横跨全部 8631 帧按类/局/split 筛选、给坏帧打标签、导出干净集重训。
  适合「**找出坏帧并剔除**」。
- **B · CVAT** — 本地标注编辑器，把框**改画**后无损写回 `labels/`。适合「**修正框的位置**」。

> 全程**本地**完成（雀魂截图不得外传，见 `docs/DESIGN.md` §7）；云端 Roboflow 等不可用。
> 命令为 **PowerShell**；每开新终端先跑一次：
> ```powershell
> $PY = "C:/Users/zsx/miniforge3/envs/auto/python.exe"
> $env:PYTHONPATH = "."
> ```
> 三个脚本都在 `scripts/inspect/`，一律**从仓库根运行**。

---

## A · FiftyOne（浏览 / 清理）

### 安装（一次性）

```powershell
& $PY -m pip install fiftyone
```

> ⚠️ **protobuf 升级**：FiftyOne 1.18 的 tagging 子系统需要 `protobuf>=5.26`，本仓环境原为 4.25.3
> （无任何包 pip 依赖它）。已升级到 **protobuf 7.x** 使 GUI 打标签可用；majsoul_eye 全部测试仍通过。
> 若日后某个 dev 抓包脚本（liqi `_pb2`）报 protobuf 相关错，回退：`& $PY -m pip install "protobuf<5"`。

### 启动 / 快速看统计

```powershell
& $PY scripts/inspect/fiftyone_view.py            # 起 GUI（浏览器打开 http://localhost:5151）
& $PY scripts/inspect/fiftyone_view.py --check     # 只构建 + 打印统计，不开界面
```

首次运行会构建一个**持久化** FiftyOne dataset（名 `majsoul_eye_detector`，约 3–5 分钟建 8631 帧），
之后再启动是**秒开**（复用已建库）。每个样本带 `game`（哪一局）、`split`（train/val）字段，
`ground_truth` 是该帧的全部检测框。

> 加载器**不走** FiftyOne 自带 YOLOv5 导入器（那个会把 repo-root 相对路径解析错），而是直接读
> `train.txt`/`val.txt`，逐帧把 `images/→labels/`、`.png→.txt` 推导出标签，再把归一化 YOLO
> `(cx,cy,w,h)` 转成 FiftyOne 的左上角 `[x,y,w,h]`。

### 清理流程（打标签 → 导出干净集 → 重训）

1. GUI 里按 `game` / `split` / 具体类别筛选（例如只看 `5pr` 这种弱类），逐帧核对框与类别。
2. 把要剔除的坏帧打上 **tag `reject`**（GUI 里选中样本 → Tag）。tag 自动存进持久库，**跨运行不丢**
   （`split`/`game` 是数据字段，不占用 tag —— tag 专留给你标 reject）。
3. 导出剔除 reject 后的干净 train/val 列表：
   ```powershell
   & $PY scripts/inspect/fiftyone_view.py --export-clean datasets/detector_clean
   ```
   生成 `datasets/detector_clean/{train.txt,val.txt}`（保留 repo-root 相对路径）。新建一份
   `data.yaml` 把 `train`/`val` 指向它们，即可用 `scripts/train/train_detector.py` 重训。

### 在磁盘上改过标签后

FiftyOne 库是磁盘标签的一份快照。若你用 CVAT（路径 B）改了 `labels/`，让 FiftyOne 从磁盘重新导入：

```powershell
& $PY scripts/inspect/fiftyone_view.py --rebuild      # 丢弃 tag，重新读盘
```

常用 flag：`--data`（默认 `datasets/detector/data.yaml`）、`--name`、`--port`、`--check`、
`--rebuild`、`--export-clean DIR`、`--reject-tag`（默认 `reject`）。

---

## B · CVAT（修正框）

三步往返；帧名被 namespace 成 `<game>__<stem>`，所以不同 game 里同名的 `000028` 不会写错位置。

### 1) 导出成 CVAT 可导入的包

```powershell
# 整局
& $PY scripts/inspect/cvat_export.py --game precise_ai_run_1 --out cvat_pkg --zip
# 多局 / 限量
& $PY scripts/inspect/cvat_export.py --game precise_session5 --game precise_session6 --out cvat_pkg --limit 200 --zip
# 一份手挑的帧列表（repo-root 相对图片路径，例如从 FiftyOne 导出的 reject 集）
& $PY scripts/inspect/cvat_export.py --frames-list bad.txt --out cvat_pkg --zip
```

产出 `cvat_pkg/`（YOLO 1.1 结构：`obj.names` / `obj.data` / `train.txt` / `obj_train_data/`）
以及 `cvat_pkg.zip`（上传用）。

### 2) 本地起 CVAT 并修框

需 **Docker Desktop**（Windows 需 WSL2）。

```powershell
git clone https://github.com/cvat-ai/cvat
cd cvat; docker compose up -d        # 打开 http://localhost:8080，创建 superuser
```

在 CVAT 里：**Create task** → 上传 `cvat_pkg.zip`（或其中的图片）→ 标签取自 `obj.names` →
修正框 → **Export task dataset** 选 **"YOLO 1.1"**。

### 3) 写回 datasets/

```powershell
& $PY scripts/inspect/cvat_import.py 你从CVAT导出的.zip --dry-run    # 先看会改哪些、几个框
& $PY scripts/inspect/cvat_import.py 你从CVAT导出的.zip              # 真正写回
```

`--dry-run` 只报告不落盘；默认只覆盖数据集里已存在的帧（`--allow-new` 才写新帧）。
`--target-root DIR` 可写进一份镜像树（`DIR/<game>/yolo/labels/...`）而非直接改 live `datasets/`，
方便先 diff 再决定。

写回后记得让 FiftyOne（路径 A）`--rebuild` 或重训检测器时用新标签。

---

## 注意事项

- **别中途 kill FiftyOne 的构建**：进程被杀在 `add_samples` 中间会留下 mongo 计数错位
  （`len()` 与实际文档数不一致）。补救：`import fiftyone as fo; fo.delete_dataset("majsoul_eye_detector")`
  后重跑一次干净构建。
- **切分永远按局/场，绝不按帧**：清理后重训时沿用 `train.txt`/`val.txt` 的既有 kyoku 切分，
  别把同一物理牌的多帧拆到两侧（会泄漏、虚高精度）。
- **合规**：所有工具本地运行；不要把雀魂截图/精灵图上传到任何云端标注服务。
