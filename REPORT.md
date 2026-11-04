# 结果记录：SmolVLA 权重编辑（无梯度训练）提升成功率

日期：2026-11-03/04　模型：官方 `lerobot/smolvla_libero`（`models/smolvla_libero_official`），GPU 推理。
方法：参照 MIT 博文 Weight-Space Programming，不使用梯度/优化器；编辑者只读模型权重、激活、模型自己的轨迹，直接改权重张量。

## 1. 成功的结果：libero_spatial task 8

任务："pick up the black bowl next to the plate and place it on the plate"（原版 LIBERO）。

**诊断（仅模型可观测量）**：dev episode 0–9 基线 5/10。失败的 5 个里有 3 个在 eef x≥0.04 处闭合夹爪，成功的 5 个全部 x≤0.02；失败 episode 的夹爪开合 5–15 次（反复重抓），成功的只有 1 次。

**编辑**：只改 1 个张量 `model.action_out_proj.bias`，在 x 动作维加一个常数偏移 s（归一化动作单位，正值使抓取点 eef x 减小）。实现：`code/make_edit_bias.py`；编辑文件 `runs/run1/edit_B25`、`edit_B35`（`manifest.json` + `edit.npz`，含 sha256）。

| 编辑 | 偏移 s | dev(0–9) | **test(100 个)** | 抓取点 x（dev 均值） |
|---|---|---|---|---|
| 基线 | 0 | 5/10 | **42/100 = 42%** | 0.015 |
| B15 | 0.15 | 5/10 | — | 0.004 |
| B25 | 0.25 | 7/10 | **64/100 = 64%** | −0.006 |
| B35 | 0.35 | 6/10 | **64/100 = 64%** | −0.012 |

test 集在选编辑时没有看过。提升约 +22 个百分点；95% Wilson 区间：基线 33–52%，B25/B35 均 54–73%；Fisher 检验 p=0.003。
B25 与 B35 无法区分。原始数据：`eval/sp8_test_*.json`、`eval/ext/`；并行脚本：`code/run_parallel.py`。


## 4. 协议与特权信息说明（如实）

- 未使用仿真器内部物体坐标、bddl 目标位置、init_states 内容；早期一版读 `body_xpos` 的代码已删除。看过 swap 版与原版 bddl 的差异（场景描述层面）和模型输入画面。
- test = init state 10–49（共 100 个 episode，见上）；dev = 0–9。
- 前向预算：按 task 单独计，博文档位 25；object_object 那次用到 50（用户放宽后上调，已记录在 PLAN.md §11）。
- 本机旧权重与官方权重哈希不同，已弃用官方之外的权重。`lerobot/smolvla_libero_plus` 已下载但未使用（相机名 camera1/camera2 需要 rename_map）。


## 6. 文件索引（均在 `/media/python/Seagate/research/smovla_astra/`）
- 计划与全过程：`PLAN.md`（§9–13）
- 代码：`code/eval_policy.py`、`forward_tool.py`、`apply_edit.py`、`make_edit_bias.py`、`make_edit_cond2.py`、`make_edit_vision.py`、`grasp_proxy.py`、`par_run.sh`
- 评测结果：`eval/sp8_test_*.json|log`、`eval/side_B35_t*.json|log`
- 编辑与前向日志：`runs/run1/`
