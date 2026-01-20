import sys
from pathlib import Path

# Caminho absoluto da raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent

# Adiciona a raiz do projeto ao PYTHONPATH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Agora os imports do projeto FUNCIONAM
import mne
from src.motifs import trans_motifs
from src.sync import motif_connectivity_matrix

DATA_DIR = Path("data/preprocessed")
eeg_files = sorted(DATA_DIR.glob("*_EO.set"))
eeg_file = eeg_files[0]

raw = mne.io.read_raw_eeglab(
    eeg_file,
    preload=True,
    verbose=False
)

print(f"Lendo arquivo: {eeg_file.name}")

raw.pick_types(eeg=True)

signal = raw.get_data()
motifs = trans_motifs(signal)

sfreq = raw.info["sfreq"]
max_lag = int(0.05 * sfreq)  # 50 ms

C = motif_connectivity_matrix(motifs, max_lag)

print(C.shape)
print(C.min(), C.max())
