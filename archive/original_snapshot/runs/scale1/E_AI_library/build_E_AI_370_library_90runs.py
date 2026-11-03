# build_E_AI_370K_library_90runs.py
from pathlib import Path
import random
import shutil
import re

ROOT = Path(r"E:\Master thesis\calvados_tea\runs\scale1\E_AI_library")
SCRATCH_ROOT = "/scratch/jpdols/calvados/runs/scale1/E_AI_library"

RESIDUES = Path(r"E:\mnt\data\tea_fixed\output\residues\residues_gamma_6p0_370K_argdefault_minus.csv")
SLURM_SRC = Path("run_1us.slurm")

N_REPS = 3
SEED = 42
TEMP = 370
GAMMA_TAG = "6p0"
RESIDUES_NAME = f"residues_gamma_{GAMMA_TAG}_{TEMP}K_argdefault_minus.csv"

PREPARE_CODE = f'''# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from argparse import ArgumentParser
from calvados.cfg import Config, Components


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()

    cwd = os.getcwd()
    sysname = args.name.strip()

    L = 50
    N_save = 1000
    N_frames = 50000
    platform = "CUDA"

    # IMPORTANT:
    # prepare.py is run from repXX folder.
    # run.py is later run from repXX/sysname folder.
    # Therefore components.yaml must point one level up to ../input.
    residues_file = os.path.join("..", "input", "{RESIDUES_NAME}")
    fasta_file = os.path.join("..", "input", "idr.fasta")

    if not os.path.exists(os.path.join(cwd, "input", "{RESIDUES_NAME}")):
        raise FileNotFoundError("Missing residues file in ./input/")
    if not os.path.exists(os.path.join(cwd, "input", "idr.fasta")):
        raise FileNotFoundError("Missing FASTA file in ./input/")

    path = os.path.join(cwd, sysname)
    os.makedirs(path, exist_ok=True)

    config = Config(
        sysname=sysname,
        box=[L, L, L],
        temp={TEMP},
        ionic=0.17,
        pH=7,
        topol="center",
        wfreq=N_save,
        steps=N_frames * N_save,
        runtime=0,
        platform=platform,
        restart="checkpoint",
        frestart="restart.chk",
        verbose=True,
    )
    config.write(path, name="config.yaml")

    components = Components(
        molecule_type="protein",
        nmol=1,
        restraint=False,
        charge_termini="both",
        fresidues=residues_file,
        ffasta=fasta_file,
    )
    components.add(name=sysname)
    components.write(path, name="components.yaml")

    print(f"[OK] Prepared {{sysname}} in {{path}}")
    print(f"     fresidues: {{residues_file}}")
    print(f"     ffasta:    {{fasta_file}}")


if __name__ == "__main__":
    main()
'''

seq = (
"IPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVGIPAVG"
)
seq = "".join(seq.split())

blocks = [seq[i:i+5] for i in range(0, len(seq), 5)]
if len(blocks[-1]) != 5:
    blocks = blocks[:-1]
seq = "".join(blocks)

print("Cleaned sequence length:", len(seq))
print("Number of blocks:", len(blocks))

ipavg = [b for b in blocks if b == "IPAVG"]
other = [b for b in blocks if b != "IPAVG"]

rng = random.Random(SEED)
architectures = {}

def wrap_fasta(name, s):
    return f">{name}\n" + "\n".join(s[i:i+80] for i in range(0, len(s), 80)) + "\n"

def add_arch(name, block_list_or_seq):
    s = "".join(block_list_or_seq) if isinstance(block_list_or_seq, list) else block_list_or_seq
    assert len(s) == len(seq)
    architectures[name] = s

for i in range(1, 11):
    b = blocks.copy()
    rng.shuffle(b)
    add_arch(f"EAI_arch{i:02d}_blockshuffle", b)

for i in range(11, 16):
    o = other.copy()
    rng.shuffle(o)
    b = o + ipavg if i % 2 == 0 else ipavg + o
    add_arch(f"EAI_arch{i:02d}_IPAVG_sidecluster", b)

for i in range(16, 21):
    o = other.copy()
    rng.shuffle(o)
    mid = len(o) // 2
    add_arch(f"EAI_arch{i:02d}_IPAVG_middlecluster", o[:mid] + ipavg + o[mid:])

for i in range(21, 26):
    o = other.copy()
    rng.shuffle(o)
    b = []
    spacing = max(1, len(o) // len(ipavg))
    ip = ipavg.copy()
    for j, block in enumerate(o):
        b.append(block)
        if ip and j % spacing == 0:
            b.append(ip.pop(0))
    b.extend(ip)
    add_arch(f"EAI_arch{i:02d}_IPAVG_evenlyspaced", b)

for i in range(26, 31):
    letters = list(seq)
    rng.shuffle(letters)
    add_arch(f"EAI_arch{i:02d}_full_AA_scramble", "".join(letters))

assert len(architectures) == 30

ROOT.mkdir(parents=True, exist_ok=True)

slurm_template = SLURM_SRC.read_text()
submit_lines = ["#!/bin/bash", "mkdir -p /scratch/jpdols/calvados/logs", ""]

for arch_name, arch_seq in architectures.items():
    for rep in range(1, N_REPS + 1):
        rep_name = f"rep{rep:02d}"

        run_dir = ROOT / arch_name / rep_name
        input_dir = run_dir / "input"
        system_dir = run_dir / arch_name

        input_dir.mkdir(parents=True, exist_ok=True)

        (input_dir / "idr.fasta").write_text(wrap_fasta(arch_name, arch_seq))
        shutil.copy2(RESIDUES, input_dir / RESIDUES_NAME)
        (run_dir / "prepare.py").write_text(PREPARE_CODE)

        scratch_system_dir = f"{SCRATCH_ROOT}/{arch_name}/{rep_name}/{arch_name}"
        job_name = f"{arch_name}_{rep_name}_{TEMP}K"

        slurm = slurm_template
        slurm = re.sub(r"#SBATCH --job-name=.*", f"#SBATCH --job-name={job_name}", slurm)
        slurm = re.sub(r"^cd .*$", f"cd {scratch_system_dir}", slurm, flags=re.MULTILINE)

        (run_dir / "run_1us.slurm").write_text(slurm)

        submit_lines.append(f"cd {scratch_system_dir}")
        submit_lines.append("sbatch ../run_1us.slurm")
        submit_lines.append("")

(ROOT / "submit_all.sh").write_text("\n".join(submit_lines) + "\n")

prepare_lines = []
for arch_name in architectures:
    for rep in range(1, N_REPS + 1):
        run_dir = ROOT / arch_name / f"rep{rep:02d}"
        prepare_lines.append(f'cd /d "{run_dir}"')
        prepare_lines.append(f'python prepare.py --name {arch_name}')
        prepare_lines.append("")

(ROOT / "prepare_all_windows.bat").write_text("\n".join(prepare_lines))

print(f"Created {len(architectures)} architectures")
print(f"Created {len(architectures) * N_REPS} replicate run folders")
print(f"Library written to: {ROOT}")
print("Next locally:")
print(f'  cd /d "{ROOT}"')
print("  prepare_all_windows.bat")
print("Then upload to DelftBlue and run submit_all.sh")