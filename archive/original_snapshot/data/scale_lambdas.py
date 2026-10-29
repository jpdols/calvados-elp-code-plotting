import pandas as pd

# ===== INPUT =====
INPUT_FILE = r"E:\Master thesis\calvados_tea\data\residues_CALVADOS2.csv"
OUTPUT_DIR = r"E:\Master thesis\calvados_tea\data\scaled"

# scaling factors
SCALES = [1.30, 1.50, 1.70, 1.90]

# hydrophobic residues (1-letter)
HYDROPHOBIC = {"A", "V", "I", "L", "M", "F", "Y", "W", "P"}

# choose: "all" or "hydrophobic"
MODE = "hydrophobic"

# =================

df = pd.read_csv(INPUT_FILE)

for scale in SCALES:
    df_new = df.copy()

    if MODE == "all":
        df_new["lambdas"] = df_new["lambdas"] * scale

    elif MODE == "hydrophobic":
        df_new["lambdas"] = df_new.apply(
            lambda row: row["lambdas"] * scale if row["one"] in HYDROPHOBIC else row["lambdas"],
            axis=1
        )

    else:
        raise ValueError("MODE must be 'all' or 'hydrophobic'")

    # optional safety clip
    df_new["lambdas"] = df_new["lambdas"].clip(lower=0.0, upper=1.2)

    output_file = f"{OUTPUT_DIR}\\residues_scaled_{scale:.2f}.csv"
    df_new.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")