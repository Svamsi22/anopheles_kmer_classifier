"""
step1_collect_all_species.py

"""

import os
import gzip
import io
import requests
import numpy as np
import pandas as pd

from config import (
    K,
    READS_PER_SAMPLE,
    LABEL_MAP,
    ALL_KMERS,
    KMER_INDEX,
    SPECIES_REGISTRY,
)

ENA_SEARCH_URL = "https://www.ebi.ac.uk/ena/portal/api/search"



def fetch_ena_runs(taxon_id: int, species_name: str) -> pd.DataFrame:

    print(f"  Querying ENA for {species_name} (taxon {taxon_id}) ...")

    params = {
        "result": "read_run",
        "query": f"tax_tree({taxon_id}) AND instrument_platform=ILLUMINA",
        "fields": "run_accession,fastq_ftp",
        "format": "json",
        "limit": 100000,
    }

    try:
        r = requests.get(ENA_SEARCH_URL, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()

    except Exception as e:
        print(f"  [ERROR] ENA query failed: {e}")
        return pd.DataFrame()

    if not data:
        print(f"  [WARNING] No runs found for {species_name}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    df = df.dropna(subset=["fastq_ftp"])
    df = df[df["fastq_ftp"].str.strip() != ""]

    
    df["fastq_url"] = "https://" + df["fastq_ftp"].str.split(";").str[0]

    df = df.rename(columns={"run_accession": "sample_id"})

    print(f"  Runs available   : {len(df)}")

    return df[["sample_id", "fastq_url"]]



def stream_fastq(url: str, max_reads: int = READS_PER_SAMPLE):

    try:

        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        gz_file = gzip.GzipFile(fileobj=response.raw)

        read_count = 0

        for line_idx, line in enumerate(io.TextIOWrapper(gz_file)):

            if line_idx % 4 == 1:

                yield line.strip()

                read_count += 1

                if read_count >= max_reads:
                    break

    except Exception as exc:

        print(f"      [WARNING] FASTQ stream failed: {exc}")




def compute_kmer_vector(url: str) -> np.ndarray:

    counts = np.zeros(len(ALL_KMERS), dtype=np.float64)

    for sequence in stream_fastq(url):

        sequence = sequence.upper()

        for i in range(len(sequence) - K + 1):

            kmer = sequence[i:i + K]

            idx = KMER_INDEX.get(kmer)

            if idx is not None:
                counts[idx] += 1

    total = counts.sum()

    if total > 0:
        counts /= total

    return counts



def collect_species(name: str, taxon_id: int, max_samples: int) -> pd.DataFrame:

    print("\n" + "=" * 55)
    print(f"  {name}  (taxon {taxon_id})  —  max {max_samples} samples")
    print("=" * 55)

    runs_df = fetch_ena_runs(taxon_id, name)

    if runs_df.empty:
        print(f"  [ERROR] No data for {name}. Skipping.")
        return pd.DataFrame()

    runs_df = (
        runs_df
        .drop_duplicates("sample_id")
        .sample(min(max_samples, len(runs_df)), random_state=42)
        .reset_index(drop=True)
    )

    print(f"  Runs selected    : {len(runs_df)}")

    rows = []

    out_path = f"data/raw/kmer_{name}.csv"
    os.makedirs("data/raw", exist_ok=True)

    for idx, row in runs_df.iterrows():

        print(f"    [{idx+1}/{len(runs_df)}] {row.sample_id}", end=" ... ", flush=True)

        vec = compute_kmer_vector(row.fastq_url)

        if vec.sum() == 0:
            print("skipped (no reads extracted)")
            continue

        rows.append({
            "sample_id": row.sample_id,
            "species": name,
            "label": LABEL_MAP[name],
            **{ALL_KMERS[i]: vec[i] for i in range(len(vec))}
        })

        print(f"done  ({len(rows)} collected so far)")

        # checkpoint save every 50 samples
        if (idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    species_df = pd.DataFrame(rows)

    species_df.to_csv(out_path, index=False)

    print(f"  Saved {len(species_df)} samples → {out_path}")

    return species_df



def main():

    print("=" * 55)
    print("  Collecting K-mer Data — All Species")
    print("=" * 55)

    for species in SPECIES_REGISTRY:

        collect_species(
            name=species["name"],
            taxon_id=species["taxon_id"],
            max_samples=species["max_samples"],
        )

    print("\n" + "=" * 55)
    print("  All species collected.")
    print("  Next step: run step2_build_dataset.py")
    print("=" * 55)


if __name__ == "__main__":
    main()