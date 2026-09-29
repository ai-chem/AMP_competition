# Method

The library is sampled from ProtGPT3-1.3B (`AI4PD/ProtGPT3-1.3B`, revision
`8bdbca30ef91fc2fd208a1f55ccee7a0990c2dab`) with a conditional LoRA adapter.
The adapter was trained on the cleaned organizer antibacterial set. At
generation time a small network maps two targets, charge at pH 7.4 and
interface hydrophobicity at pH 8, onto soft-prompt tokens. Sampling uses the
five points in `configs/selected_generation_conditions.csv`, seed 42,
temperature 0.8, and top-p 0.9.

Each point contributes 50,000 peptides that already satisfy the competition
alphabet and length, read N-to-C, and are not exact copies of
`data/external/antibacterial.fasta`. A LightGBM ranker scores predicted MIC
from ESM-2 8M embeddings plus physicochemical descriptors. Hemolysis is an
HC50 in µM from a fine-tuned ESM-2 35M model. The sort key is the sum of the
two average ranks (higher MIC rank and higher log-HC50 rank are better).
Peptides whose `Levenshtein.ratio` to the organizer reference is above 0.80
are removed. `generate/library.fasta` is the first 50,000 survivors and
`generate/top.fasta` is the first 100 of that list.

The entry point is `uv run generate`. It expects a CUDA GPU. Base weights that
are not stored in this repository are downloaded from Hugging Face at the
pinned revisions.
