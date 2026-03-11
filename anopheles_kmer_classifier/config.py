"""
config.py

Data sources (all public, no authentication required)
-----------------------------------------------------
  Ag3    (A. gambiae)   : ENA taxonomy search  tax_eq(7165)
  Af1    (A. funestus)  : ENA taxonomy search  tax_eq(62324)
  Amin1  (A. minimus)   : ENA taxonomy search  tax_eq(7141)
  Astep  (A. stephensi) : ENA taxonomy search  tax_eq(30069)
"""

import itertools



K              = 6          # k-mer length (6-mers → 4^6 = 4,096 features)
READS_PER_SAMPLE = 10_000   # FASTQ reads streamed per sample



SAMPLES_PER_SPECIES = 500   


TRAIN_PER_SPECIES = 300
VAL_PER_SPECIES   = 100
TEST_PER_SPECIES  = 100




SPECIES_REGISTRY = [
    {"name": "Astep", "taxon_id": 30069, "max_samples": SAMPLES_PER_SPECIES},
    {"name": "Ag3",   "taxon_id": 7165,  "max_samples": SAMPLES_PER_SPECIES},
    {"name": "Af1",   "taxon_id": 62324,  "max_samples": SAMPLES_PER_SPECIES},
]



LABEL_MAP = {
    "Ag3":   0,   # Anopheles gambiae
    "Af1":   1,   # Anopheles funestus
    "Astep": 2,   # Anopheles stephensi
}

SPECIES_NAMES = list(LABEL_MAP.keys())   # ordered list for plotting



SPECIES_COLORS = {
    "Ag3":   "steelblue",
    "Af1":   "orange",
    "Astep": "mediumpurple",
}



# Raw outputs from data-collection steps
RAW_AG3   = "data/raw/kmer_Ag3.csv"
RAW_AF1   = "data/raw/kmer_Af1.csv"
RAW_ASTEP = "data/raw/kmer_Astep.csv"

# After balancing Ag3/Af1
BALANCED_TRAIN = "data/processed/kmer_train_balanced.csv"
BALANCED_VAL   = "data/processed/kmer_validation_balanced.csv"
BALANCED_TEST  = "data/processed/kmer_test_balanced.csv"

# After merging in Amin1 → final inputs for modelling
FINAL_TRAIN = "data/final/kmer_train_final.csv"
FINAL_VAL   = "data/final/kmer_validation_final.csv"
FINAL_TEST  = "data/final/kmer_test_final.csv"

# Output figures
FIG_PCA        = "outputs/figures/pca_clustering.png"
FIG_TSNE       = "outputs/figures/tsne_clustering.png"
FIG_CONFUSION  = "outputs/figures/confusion_matrix.png"
FIG_COEF       = "outputs/figures/kmer_coefficients.png"
FIG_COMPARISON = "outputs/figures/baseline_comparison.png"



BASES     = ["A", "C", "G", "T"]
ALL_KMERS = ["".join(p) for p in itertools.product(BASES, repeat=K)]
KMER_INDEX = {kmer: idx for idx, kmer in enumerate(ALL_KMERS)}



META_COLS = ["sample_id", "species", "label", "split"]
