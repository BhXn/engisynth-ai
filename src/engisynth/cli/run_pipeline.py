import argparse, pandas as pd, yaml, os
from engisynth.config.load import load_config
from engisynth.data.io import read_table
from engisynth.generators.ctgan import CTGANAdapter
from engisynth.evaluation.metrics import delta_r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    df = read_table(args.csv)
    gen = CTGANAdapter(epochs=cfg.generator.epochs, batch_size=cfg.generator.batch_size)
    gen.fit(df)
    synth = gen.sample(len(df))
    dr = delta_r(df, synth, cfg.target, task="classification")
    os.makedirs(args.output, exist_ok=True)
    df.to_csv(f"{args.output}/original.csv", index=False)
    synth.to_csv(f"{args.output}/synthetic.csv", index=False)
    with open(f"{args.output}/report.txt", "w") as f:
        f.write(f"ΔR = {dr:.4f}\n")
    print("Pipeline done, ΔR =", dr)

if __name__ == "__main__":
    main()

