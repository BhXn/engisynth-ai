#!/usr/bin/env python
"""
批量下载 OpenML Task 数据，
生成 CSV、自动写初版 config，
跑一次 CTGAN+LightGBM pipeline，
并汇总 ΔR 到 summary.csv。
"""
import os
import yaml
import pandas as pd
import openml
from subprocess import check_call
from engisynth.cli.run_pipeline import run_pipeline

def main():
    # 1. 读配置
    with open("configs/datasets.yml", "r") as f:
        cfg = yaml.safe_load(f)
    candidates = cfg["candidates"]

    results = []
    os.makedirs("outputs/eda", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    for item in candidates:
        name = item["name"]
        task_id = item["openml_task_id"]
        target = item["target_name"]

        print(f"\n=== Processing {name} (Task {task_id}) ===")

        # 2. 下载并拼 DataFrame
        task = openml.tasks.get_task(task_id)
        X, y, _, _ = task.get_dataset().get_data(
            target=target,
            dataset_format="dataframe"
        )
        df = pd.concat([X, y.rename(target)], axis=1)

        # 3. 写 CSV
        csv_path = f"data/{name}.csv"
        df.to_csv(csv_path, index=False)
        print("  → Saved CSV:", csv_path)

        # 4. 生成初版 config
        cfg_path = f"configs/{name}.yaml"
        cmd_cfg = [
            "python", "src/engisynth/cli/create_config.py",
            "--csv", csv_path, "--out", cfg_path
        ]
        print("  → Generating config:", " ".join(cmd_cfg))
        check_call(cmd_cfg)

        # 5. 跑 pipeline
        out_dir = f"outputs/eda/{name}"
        print("  → Running pipeline, output:", out_dir)
        run_pipeline(csv=csv_path, cfg=cfg_path, out=out_dir)

        # 6. 读取 ΔR
        rpt = pd.read_json(os.path.join(out_dir, "report.json"), typ="series")
        delta_r = rpt.get("delta_R", None)
        print(f"  → {name} ΔR = {delta_r:.4f}")

        results.append({
            "dataset": name,
            "task": item["task"],
            "delta_R": delta_r
        })

    # 7. 汇总结果
    summary = pd.DataFrame(results)
    summary_path = "outputs/eda/summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nAll done. Summary:\n", summary)
    print("Written to", summary_path)


if __name__ == "__main__":
    main()
