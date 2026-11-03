# -*- coding: utf-8 -*-
"""
CALVADOS single-chain run preparation (Windows-safe)

Creates:
  <sysname>/config.yaml
  <sysname>/components.yaml
  <sysname>/run.py

Usage (from this folder):
  python prepare.py --name <FASTA header>

Expected inputs in ./input:
  - residues_CALVADOS2.csv
  - idr.fasta   (FASTA header must match --name)
"""

from __future__ import annotations

import os
from argparse import ArgumentParser

from calvados.cfg import Config, Components


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--name", required=True, type=str, help="System/component name (must match FASTA header)")
    args = parser.parse_args()

    cwd = os.getcwd()
    sysname = args.name.strip()

    # ----------------------------
    # Quick sanity-run settings
    # ----------------------------
    L = 50          # nm, cubic box side length (single chain: plenty)
    N_save = 1000   # write interval (integration steps)
    N_frames = 50000   # number of frames -> total steps = N_frames * N_save =

    # Switch to 'CUDA' later for speed (if available)
    platform = "CUDA"  # "CPU" or "CUDA"

    # ----------------------------
    # Input files (relative to this run folder)
    # ----------------------------
    residues_file = os.path.join(
        cwd, "input", "residues_gamma_6p0_370K_argdefault_minus.csv"
    )
    fasta_file = os.path.join(cwd, "input", "idr.fasta")

    if not os.path.exists(residues_file):
        raise FileNotFoundError(f"Missing residues file: {residues_file}")
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"Missing FASTA file: {fasta_file}")

    # ----------------------------
    # Output folder(s)
    # ----------------------------
    path = os.path.join(cwd, sysname)
    os.makedirs(path, exist_ok=True)

    # ----------------------------
    # CALVADOS config
    # ----------------------------
    config = Config(
        # GENERAL
        sysname=sysname,
        box=[L, L, L],          # nm
        temp=370,            # K
        ionic=0.17,             # M
        pH=7,
        topol="center",

        # RUNTIME SETTINGS
        wfreq=N_save,           # dcd writing interval, 1 = 10 fs
        steps=N_frames * N_save,
        runtime=0,              # if >0 overwrites steps
        platform=platform,      # "CPU" or "CUDA"
        restart="checkpoint",
        frestart="restart.chk",
        verbose=True,
    )

    config.write(path, name="config.yaml")

    # ----------------------------
    # Components: single protein chain from FASTA
    # ----------------------------
    components = Components(
        molecule_type="protein",
        nmol=1,
        restraint=False,
        charge_termini="both",
        fresidues=residues_file,
        ffasta=fasta_file,
    )

    # IMPORTANT: name must match FASTA header (line after ">")
    components.add(name=sysname)
    components.write(path, name="components.yaml")

    print(f"[OK] Prepared system '{sysname}' in: {path}")
    print(f"     Inputs: {fasta_file}  |  {residues_file}")
    print(f"     Steps: {N_frames * N_save}  (N_frames={N_frames}, N_save={N_save})")
    print(f"     Platform: {platform}")
    print("Next:")
    print(f'  python {sysname}/run.py --path {sysname}')


if __name__ == "__main__":
    main()
