import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import data as tm_data


def flush_chunks(chunks, out_path):
    if not chunks:
        return
    pd.concat(chunks, ignore_index=True).to_hdf(
        out_path, key="trips", mode="a", format="table", append=True, index=False
    )
    chunks.clear()


def build_windowed_cache(input_pkl, output_h5, compressed_output_h5, city, chunk_rows):
    input_pkl = Path(input_pkl)
    output_h5 = Path(output_h5)
    compressed_output_h5 = Path(compressed_output_h5)

    for path in [output_h5, compressed_output_h5]:
        if path.exists():
            path.unlink()

    df = pd.read_pickle(input_pkl)
    chunks = []
    rows_in_chunks = 0
    stats = {"rows": 0, "kept": 0, "split": 0, "dropped": 0, "windows": 0}

    for row_i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Building windowed train")):
        tid = getattr(row, "tid")
        lats = list(getattr(row, "sparse_lat_list"))
        lngs = list(getattr(row, "sparse_lng_list"))
        timestamps = list(getattr(row, "sparse_time_list"))
        roads = list(getattr(row, "sparse_gps2route_list"))
        n = min(len(lats), len(lngs), len(timestamps), len(roads))

        if n < tm_data.MIN_TRIP_LEN:
            stats["dropped"] += 1
            continue

        # Include row_i because tid can repeat in the exported pkl.
        unique_tid = f"{tid}__row{row_i:06d}"
        base = pd.DataFrame(
            {
                tm_data.TRAJ_ID_COL: [unique_tid] * n,
                tm_data.X_COL: lngs[:n],
                tm_data.Y_COL: lats[:n],
                tm_data.T_COL: timestamps[:n],
                tm_data.ROAD_COL: roads[:n],
            }
        )

        if base.isna().any().any():
            stats["dropped"] += 1
            continue

        windows = tm_data.DataPreprocessor.split_trip_into_windows(base, unique_tid)
        stats["split" if len(windows) > 1 else "kept"] += 1

        for window in windows:
            window = tm_data.DataPreprocessor.cal_high_order_features(window)
            chunks.append(window)
            rows_in_chunks += len(window)
            stats["windows"] += 1
            stats["rows"] += len(window)

        if rows_in_chunks >= chunk_rows:
            flush_chunks(chunks, output_h5)
            rows_in_chunks = 0

    flush_chunks(chunks, output_h5)
    print("windowed_stats", stats)
    print("wrote", output_h5, output_h5.stat().st_size)

    tm_data.setting = {"dataset": {"city": city}}
    preprocessor = tm_data.DataPreprocessor(city)
    suffix = output_h5.stem.replace(f"{city}_train", "")
    preprocessor.compress_traj(dataset_mode="train", pred_len=0, input_suffix=suffix, output_suffix=suffix)
    print("wrote", compressed_output_h5, compressed_output_h5.stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", default="/root/autodl-tmp/xian/xian_train_extension65_50_35_road2gps.pkl")
    parser.add_argument("--city", default="xian")
    parser.add_argument("--suffix", default="_windowed")
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    args = parser.parse_args()

    out_dir = Path(tm_data.TRAJ_META_DIR)
    output_h5 = out_dir / f"{args.city}_train{args.suffix}.h5"
    compressed_output_h5 = out_dir / f"{args.city}_train{args.suffix}_compressed_keep-0.h5"
    build_windowed_cache(args.input_pkl, output_h5, compressed_output_h5, args.city, args.chunk_rows)


if __name__ == "__main__":
    sys.exit(main())
