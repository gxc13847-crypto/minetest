#!/usr/bin/env python3
"""Prepare TrajMamba native similar-trajectory search test artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "tools" / "textual_embedding"))

from data import DataPreprocessor, TRAJ_META_DIR, SEARCH_META_DIR  # noqa: E402
from export_trajmamba_inputs import build_trips_df  # noqa: E402


def export_eval_subset(city: str, pkl_path: Path, max_samples: int, seed: int) -> Path:
    source = pd.read_pickle(pkl_path)
    if len(source) > max_samples:
        source = source.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    subset_pkl = Path(TRAJ_META_DIR) / f"{city}_eval_subset.pkl"
    source.to_pickle(subset_pkl)
    trips_df, skipped = build_trips_df(
        subset_pkl,
        route_col="sparse_gps2route_list",
        min_len=5,
        max_len=120,
        coord_source="sparse",
        time_source="sparse",
        window_long_trips=True,
        window_stride=60,
    )
    test_h5 = Path(TRAJ_META_DIR) / f"{city}_test.h5"
    test_h5.parent.mkdir(parents=True, exist_ok=True)
    trips_df.to_hdf(test_h5, key="trips", mode="w")

    print(f"[{city}] exported {trips_df['trip'].nunique()} trips, {len(trips_df)} points -> {test_h5}")
    print(f"[{city}] skipped={dict(skipped)}")
    return test_h5


def build_search_meta(city: str, search_params: dict) -> None:
    preprocessor = DataPreprocessor(city)
    preprocessor.compress_traj(dataset_mode="test", pred_len=0)
    preprocessor.Testset_SimTraj_Label(**search_params)
    preprocessor.construct_STS_meta(
        f"{city}_test_compressed_keep-0",
        gen_indices=True,
        only_indices=False,
        **search_params,
    )
    print(f"[{city}] search meta ready under {SEARCH_META_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["xian", "chengdu"])
    parser.add_argument("--pkl", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-target", type=int, default=200)
    parser.add_argument("--num-negative", type=int, default=500)
    parser.add_argument("--same-od-thres", type=int, default=50)
    args = parser.parse_args()

    search_params = {
        "num_target": args.num_target,
        "num_negative": args.num_negative,
        "same_OD_thres": args.same_od_thres,
    }
    export_eval_subset(args.city, args.pkl, args.max_samples, args.seed)
    build_search_meta(args.city, search_params)


if __name__ == "__main__":
    main()
