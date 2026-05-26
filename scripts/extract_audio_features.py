"""
Pipeline di estrazione feature audio per vocalizzazioni gallina.

Gestisce:
1. Normalizzazione nomi file anomali
2. Preprocessing audio (mono, resample)
3. Estrazione feature acustiche (basic, spettrali, MFCC, pitch)
4. Aggregazione a livello gallina (una riga per file audio)
5. Salvataggio Parquet + CSV
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal
from scipy.stats import percentileofscore
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
AUDIOS_DIR = WORKSPACE_ROOT / "data" / "raw" / "audios"
OUTPUT_DIR = WORKSPACE_ROOT / "data" / "processed" / "audio_features_output"

SR = 22050  # Sample rate di riferimento
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 20

# Parametri per segmentazione eventi (blocchi di 3s con hop 1.5s)
BLOCK_DURATION = 3.0
BLOCK_HOP = 1.5
SILENCE_THRESHOLD_DB = -40  # Soglia silenzio relativa al RMS

# ============================================================================
# UTILITY: Normalizzazione nomi file
# ============================================================================

def normalize_audio_filename(filename: str) -> Tuple[str, Optional[str]]:
    """
    Normalizza nomi file anomali: trim spazi, validazione pattern ID.
    
    Returns:
        (normalized_name, error_message)
        Se error_message è None, il file è valido e rinominabile.
    """
    # Rimuovi spazi all'inizio/fine e .WAV
    base = filename.strip()
    if base.upper().endswith('.WAV'):
        base = base[:-4]
    
    # Normalizza estensione
    base_clean = base.strip()
    
    # Valida pattern: Rxx-yy o RE37-18
    # Pattern standard: R + 2 digit + - + 2 digit
    pattern_standard = r'^R(\d{2})-(\d{2})$'
    pattern_re = r'^RE(\d{2})-(\d{2})$'
    
    match_std = re.match(pattern_standard, base_clean, re.IGNORECASE)
    match_re = re.match(pattern_re, base_clean, re.IGNORECASE)
    
    if match_std:
        # Pattern standard: normalizza case
        r_part = match_std.group(1)
        yy_part = match_std.group(2)
        normalized = f"R{r_part}-{yy_part}.WAV"
        return normalized, None
    elif match_re:
        # Pattern RE: normalizza case
        r_part = match_re.group(1)
        yy_part = match_re.group(2)
        normalized = f"RE{r_part}-{yy_part}.WAV"
        return normalized, None
    else:
        return None, f"Pattern non riconosciuto: {base_clean}"

def rename_anomalous_files(audios_dir: Path, dry_run: bool = False) -> Dict:
    """
    Identifica file anomali e li rinomina in modo deterministico.
    
    Returns:
        {
            'total_files': int,
            'renamed': List[Tuple[old, new]],
            'skipped_invalid': List[Tuple[filename, reason]],
            'collisions': List[Tuple[file1, file2, normalized_name]],
        }
    """
    result = {
        'total_files': 0,
        'renamed': [],
        'skipped_invalid': [],
        'collisions': []
    }
    
    files = list(audios_dir.glob('*.WAV')) + list(audios_dir.glob('*.wav'))
    result['total_files'] = len(files)
    
    # Mappa di tutti i nomi normalizzati per rilevare collisioni
    normalized_to_original = {}
    renaming_plan = {}  # old_path -> new_path
    
    for fpath in sorted(files):
        old_name = fpath.name
        new_name, error = normalize_audio_filename(old_name)
        
        if error:
            result['skipped_invalid'].append((old_name, error))
            continue
        
        # Controlla collisioni
        if new_name in normalized_to_original:
            result['collisions'].append((
                normalized_to_original[new_name],
                old_name,
                new_name
            ))
            continue
        
        normalized_to_original[new_name] = old_name
        
        # Se il nome è già corretto, salta
        if old_name == new_name:
            continue
        
        # Altrimenti pianifica rinomina
        new_path = audios_dir / new_name
        renaming_plan[str(fpath)] = str(new_path)
    
    # Esegui rinomina (salvo se dry_run=True)
    for old_path_str, new_path_str in renaming_plan.items():
        old_path = Path(old_path_str)
        new_path = Path(new_path_str)
        
        if not dry_run:
            old_path.rename(new_path)
        
        result['renamed'].append((old_path.name, new_path.name))
    
    return result

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_basic_features(y: np.ndarray, sr: int, duration_sec: float) -> Dict[str, float]:
    """
    Estrai feature basic: RMS, silenzio %, eventi ad alta energia, ecc.
    """
    features = {}
    
    # RMS energy direttamente dal segnale
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    features['rms_max'] = float(np.max(rms))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))
    
    # Silenzio (frame sotto soglia)
    rms_db = librosa.power_to_db(rms ** 2)
    rms_db_max = np.max(rms_db)
    silence_threshold = rms_db_max + SILENCE_THRESHOLD_DB
    silence_frames = np.sum(rms_db < silence_threshold)
    features['silence_percent'] = float(100.0 * silence_frames / len(rms_db))
    
    # Numero di "burst" ad alta energia (picchi nella RMS)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(rms, height=np.mean(rms) + np.std(rms), distance=int(sr / HOP_LENGTH * 0.1))
    features['num_high_energy_events'] = len(peaks)
    features['vocalization_rate_per_min'] = float(len(peaks) * 60 / duration_sec)
    
    # Durata media degli eventi
    if len(peaks) > 1:
        peak_distances = np.diff(peaks)
        features['mean_event_duration_sec'] = float(np.mean(peak_distances) * HOP_LENGTH / sr)
    else:
        features['mean_event_duration_sec'] = 0.0
    
    return features

def extract_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Estrai feature spettrali (centroid, bandwidth, rolloff, flatness, ZCR).
    """
    features = {}
    
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    features['spectral_centroid_mean'] = float(np.mean(centroid))
    features['spectral_centroid_std'] = float(np.std(centroid))
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
    features['spectral_bandwidth_std'] = float(np.std(bandwidth))
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    features['spectral_rolloff_mean'] = float(np.mean(rolloff))
    features['spectral_rolloff_std'] = float(np.std(rolloff))
    
    # Spectral flatness
    D = librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    S = np.abs(D) ** 2
    # Flatness per frame: media geometrica / media aritmetica
    eps = 1e-10
    flatness = np.zeros(S.shape[1])
    for i in range(S.shape[1]):
        gmean = np.exp(np.mean(np.log(S[:, i] + eps)))
        amean = np.mean(S[:, i])
        flatness[i] = gmean / (amean + eps)
    features['spectral_flatness_mean'] = float(np.mean(flatness))
    features['spectral_flatness_std'] = float(np.std(flatness))
    
    return features

def extract_mfcc_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Estrai MFCC (20 coefficienti) aggregati con statistiche (mean, std, min, max, median, p10, p90).
    """
    features = {}
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    for coeff_idx in range(N_MFCC):
        coeff_data = mfcc[coeff_idx]
        prefix = f"mfcc_{coeff_idx+1:02d}"
        
        features[f"{prefix}_mean"] = float(np.mean(coeff_data))
        features[f"{prefix}_std"] = float(np.std(coeff_data))
        features[f"{prefix}_min"] = float(np.min(coeff_data))
        features[f"{prefix}_max"] = float(np.max(coeff_data))
        features[f"{prefix}_median"] = float(np.median(coeff_data))
        features[f"{prefix}_p10"] = float(np.percentile(coeff_data, 10))
        features[f"{prefix}_p90"] = float(np.percentile(coeff_data, 90))
    
    return features

def extract_pitch_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Estrai feature di pitch usando librosa piptrack (robusto).
    Se il pitch non è affidabile, ritorna NaN + quality flag.
    """
    features = {}
    
    try:
        # Stima f0 con piptrack
        f0 = librosa.yin(y, fmin=50, fmax=2000, sr=sr)
        
        # Filtra f0 non validi (0 significa voiceless)
        f0_valid = f0[f0 > 0]
        
        if len(f0_valid) > 0:
            features['f0_mean'] = float(np.mean(f0_valid))
            features['f0_std'] = float(np.std(f0_valid))
            features['f0_min'] = float(np.min(f0_valid))
            features['f0_max'] = float(np.max(f0_valid))
            features['f0_valid_ratio'] = float(len(f0_valid) / len(f0))
        else:
            # Nessun pitch affidabile rilevato
            features['f0_mean'] = np.nan
            features['f0_std'] = np.nan
            features['f0_min'] = np.nan
            features['f0_max'] = np.nan
            features['f0_valid_ratio'] = 0.0
    except Exception as e:
        # Se l'estrazione fallisce, mark as unavailable
        features['f0_mean'] = np.nan
        features['f0_std'] = np.nan
        features['f0_min'] = np.nan
        features['f0_max'] = np.nan
        features['f0_valid_ratio'] = 0.0
    
    return features

def process_audio_file(audio_path: Path) -> Optional[Dict]:
    """
    Carica audio, lo preprocessa e estrae tutte le feature.
    Ritorna un dizionario con l'ID e tutte le feature, o None se fail.
    """
    try:
        # Estrai ID dal nome file (senza estensione)
        chicken_id = audio_path.stem  # es. "R01-03"
        
        # Carica audio al sample rate target per evitare problemi FFT
        y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
        duration_sec = librosa.get_duration(y=y, sr=sr)
        
        # Normalizzazione ampiezza (peak-safe)
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        # Estrai feature
        result = {
            'chicken_id': chicken_id,
            'duration_sec': duration_sec,
            'sample_rate': sr,
        }
        
        result.update(extract_basic_features(y, sr, duration_sec))
        result.update(extract_spectral_features(y, sr))
        result.update(extract_mfcc_features(y, sr))
        result.update(extract_pitch_features(y, sr))
        
        return result
    
    except Exception as e:
        print(f"  ⚠ Errore processamento {audio_path.name}: {e}")
        return None

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("PIPELINE ESTRAZIONE FEATURE AUDIO GALLINA")
    print("=" * 80)
    print(f"Data/Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workspace: {WORKSPACE_ROOT}")
    print(f"Audios dir: {AUDIOS_DIR}")
    print()
    
    # Step 1: Crea output dir
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Step 2: Rinomina file anomali
    print("STEP 1: Normalizzazione nomi file")
    print("-" * 80)
    rename_result = rename_anomalous_files(AUDIOS_DIR, dry_run=False)
    print(f"  File totali: {rename_result['total_files']}")
    print(f"  Rinominati: {len(rename_result['renamed'])}")
    if rename_result['renamed']:
        for old, new in rename_result['renamed'][:5]:
            print(f"    {old} → {new}")
        if len(rename_result['renamed']) > 5:
            print(f"    ... e {len(rename_result['renamed']) - 5} altri")
    print(f"  Scartati (pattern invalido): {len(rename_result['skipped_invalid'])}")
    if rename_result['skipped_invalid']:
        for fname, reason in rename_result['skipped_invalid'][:3]:
            print(f"    {fname}: {reason}")
    if rename_result['collisions']:
        print(f"  ⚠ Collisioni rilevate: {len(rename_result['collisions'])}")
        for file1, file2, normalized in rename_result['collisions'][:3]:
            print(f"    {file1} e {file2} → {normalized}")
    print()
    
    # Step 3: Inventario file validi dopo rinomina
    print("STEP 2: Inventario file audio validi")
    print("-" * 80)
    audio_files = sorted(AUDIOS_DIR.glob('*.WAV'))
    print(f"  File WAV trovati: {len(audio_files)}")
    if len(audio_files) == 0:
        print("  ⚠ Nessun file WAV trovato!")
        return
    print()
    
    # Step 4: Estrai feature per ogni file
    print("STEP 3: Estrazione feature audio")
    print("-" * 80)
    all_features = []
    for audio_path in tqdm(audio_files, desc="Processing audio"):
        feature_dict = process_audio_file(audio_path)
        if feature_dict is not None:
            all_features.append(feature_dict)
    
    print(f"  File processati con successo: {len(all_features)}/{len(audio_files)}")
    print()
    
    if len(all_features) == 0:
        print("  ⚠ Nessun file è stato processato con successo!")
        return
    
    # Step 5: Costruisci tabella
    print("STEP 4: Costruzione tabella feature")
    print("-" * 80)
    df = pd.DataFrame(all_features)
    print(f"  Shape: {df.shape}")
    print(f"  Colonne: {df.shape[1]}")
    print(f"  Righe (uno per gallina): {df.shape[0]}")
    print(f"  Colonne: {list(df.columns)[:10]}... (e {df.shape[1] - 10} altre)")
    print()
    
    # Step 6: Salvataggio Parquet
    print("STEP 5: Salvataggio output")
    print("-" * 80)
    
    # Parquet (master)
    parquet_path = OUTPUT_DIR / "audio_features.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  ✓ Parquet salvato: {parquet_path}")
    print(f"    Dimensione: {parquet_path.stat().st_size / (1024**2):.2f} MB")
    
    # CSV (interoperabilità)
    csv_path = OUTPUT_DIR / "audio_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV salvato: {csv_path}")
    print(f"    Dimensione: {csv_path.stat().st_size / (1024**2):.2f} MB")
    
    # Metadata di estrazione
    metadata = {
        'extraction_timestamp': datetime.now().isoformat(),
        'workspace': str(WORKSPACE_ROOT),
        'total_audio_files_found': len(audio_files),
        'successfully_processed': len(all_features),
        'failed': len(audio_files) - len(all_features),
        'parameters': {
            'sample_rate': SR,
            'frame_length': FRAME_LENGTH,
            'hop_length': HOP_LENGTH,
            'n_mfcc': N_MFCC,
            'block_duration_sec': BLOCK_DURATION,
            'block_hop_sec': BLOCK_HOP,
            'silence_threshold_db': SILENCE_THRESHOLD_DB,
        },
        'renaming_summary': {
            'total_files': rename_result['total_files'],
            'renamed': len(rename_result['renamed']),
            'invalid_pattern': len(rename_result['skipped_invalid']),
            'collisions': len(rename_result['collisions']),
        }
    }
    
    metadata_path = OUTPUT_DIR / "extraction_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata salvato: {metadata_path}")
    
    # Step 7: Statistiche basiche
    print()
    print("STEP 6: Statistiche descrittive (sample)")
    print("-" * 80)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().round(3).to_string())
    
    print()
    print("=" * 80)
    print("✓ PIPELINE COMPLETATO CON SUCCESSO")
    print("=" * 80)

if __name__ == "__main__":
    main()
