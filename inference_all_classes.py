import os
import sys
import csv
import math
import torch
import uproot
import numpy as np
from tqdm import tqdm
from math import cos, sin, sinh
from pathlib import Path

# path setup
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from networks.example_ParticleTransformer_sophon import get_model

# CONFIGURATION
TARGET_EVENTS_PER_CLASS = 100_000  # Events to process per class (set to None for all)
MAX_PART = 128
STEP_SIZE = 5000
TREE_NAME = "tree"
ROOT_DIR = "val_5M"
OUTPUT_DIR = "embeddings"  # Output directory for CSVs

# Skip if output file already exists
SKIP_IF_EXISTS = True

# ALL JET CLASSES
JET_CLASSES = {
    "HToBB": {
        "files": ["HToBB_120.root", "HToBB_121.root", "HToBB_122.root", "HToBB_123.root", "HToBB_124.root"],
        "output": "HToBB_inference_with_embedding.csv"
    },
    "HToCC": {
        "files": ["HToCC_120.root", "HToCC_121.root", "HToCC_122.root", "HToCC_123.root", "HToCC_124.root"],
        "output": "HToCC_inference_with_embedding.csv"
    },
    "HToGG": {
        "files": ["HToGG_120.root", "HToGG_121.root", "HToGG_122.root", "HToGG_123.root", "HToGG_124.root"],
        "output": "HToGG_inference_with_embedding.csv"
    },
    "HToWW4Q": {
        "files": ["HToWW4Q_120.root", "HToWW4Q_121.root", "HToWW4Q_122.root", "HToWW4Q_123.root", "HToWW4Q_124.root"],
        "output": "HToWW4Q_inference_with_embedding.csv"
    },
    "HToWW2Q1L": {
        "files": ["HToWW2Q1L_120.root", "HToWW2Q1L_121.root", "HToWW2Q1L_122.root", "HToWW2Q1L_123.root", "HToWW2Q1L_124.root"],
        "output": "HToWW2Q1L_inference_with_embedding.csv"
    },
    "ZToQQ": {
        "files": ["ZToQQ_120.root", "ZToQQ_121.root", "ZToQQ_122.root", "ZToQQ_123.root", "ZToQQ_124.root"],
        "output": "ZToQQ_inference_with_embedding.csv"
    },
    "WToQQ": {
        "files": ["WToQQ_120.root", "WToQQ_121.root", "WToQQ_122.root", "WToQQ_123.root", "WToQQ_124.root"],
        "output": "WToQQ_inference_with_embedding.csv"
    },
    "TTBar": {
        "files": ["TTBar_120.root", "TTBar_121.root", "TTBar_122.root", "TTBar_123.root", "TTBar_124.root"],
        "output": "TTBar_inference_with_embedding.csv"
    },
    "TTBarLep": {
        "files": ["TTBarLep_120.root", "TTBarLep_121.root", "TTBarLep_122.root", "TTBarLep_123.root", "TTBarLep_124.root"],
        "output": "TTBarLep_inference_with_embedding.csv"
    },
    "ZJetsToNuNu": {
        "files": ["ZJetsToNuNu_120.root", "ZJetsToNuNu_121.root", "ZJetsToNuNu_122.root", "ZJetsToNuNu_123.root", "ZJetsToNuNu_124.root"],
        "output": "ZToNuNu_inference_with_embedding.csv"
    },
}

# KEYS
particle_keys = [
    'part_px', 'part_py', 'part_pz', 'part_energy',
    'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
    'part_dzval', 'part_dzerr', 'part_charge',
    'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon'
]
#Labels removed from model input to prevent data leakage
# still read for get_truth_label() but not fed to the model
scalar_keys_for_model = [
    'jet_pt','jet_eta','jet_phi',
    'jet_energy','jet_nparticles','jet_sdmass','jet_tau1',
    'jet_tau2','jet_tau3','jet_tau4','aux_genpart_eta',
    'aux_genpart_phi','aux_genpart_pid','aux_genpart_pt',
    'aux_truth_match'
]
# Labels needed for ground truth extraction (not fed to model)
label_keys = [
    'label_QCD','label_Hbb','label_Hcc','label_Hgg',
    'label_H4q','label_Hqql','label_Zqq','label_Wqq',
    'label_Tbqq','label_Tbl'
]
scalar_keys = label_keys + scalar_keys_for_model
pf_keys = particle_keys + scalar_keys

label_names = ["QCD","Hbb","Hcc","Hgg","Htoww4q","Hqql","Zqq","Wqq","Htoww2q1L","Tbqq", "Tbl"]


# MODEL SETUP
class DummyDataConfig:
    input_dicts = {"pf_features": list(range(37))}
    input_names = ["pf_points"]
    input_shapes = {"pf_points": (MAX_PART, 37)}
    label_names = ["label"]
    num_classes = 10

data_config = DummyDataConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model on {device}...")
model, _ = get_model(data_config, num_classes=data_config.num_classes, export_embed=True)
model.eval().to(device)
print("Model loaded")


# HELPER FUNCTIONS
def build_pf_tensor(arrays, i):
    """Return model inputs for event i, or None if too many particles."""
    n_part = arrays["part_px"][i].shape[0]
    if n_part > MAX_PART:
        return None
    particle_feats = [arrays[k][i] for k in particle_keys]
    # Use scalar_keys_for_model (NOT scalar_keys) to exclude labels from model input
    scalar_feats = [np.full(n_part, arrays[k][i]) for k in scalar_keys_for_model]
    all_feats = particle_feats + scalar_feats
    pf_features = np.stack(all_feats, axis=1).astype(np.float32)
    padded = np.zeros((MAX_PART, pf_features.shape[1]), dtype=np.float32)
    padded[:n_part, :] = pf_features
    jet_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
    lorentz_vectors = jet_tensor[:, :, 0:4].transpose(1, 2)
    features = jet_tensor[:, :, 4:].transpose(1, 2)
    mask = (jet_tensor.sum(dim=2) != 0).unsqueeze(1)
    points = None
    return points, features, lorentz_vectors, mask

def get_truth_label(arrays, i):
    labs = np.array([arrays[k][i] for k in [
        'label_QCD','label_Hbb','label_Hcc','label_Hgg',
        'label_H4q','label_Hqql','label_Zqq','label_Wqq',
        'label_Tbqq','label_Tbl'
    ]])
    y = int(np.argmax(labs))
    return y, label_names[y] if y < len(label_names) else "Unknown"

def jet_masses(arrays, i):
    jet_sdmass = float(arrays["jet_sdmass"][i])
    pt  = float(arrays["jet_pt"][i])
    eta = float(arrays["jet_eta"][i])
    phi = float(arrays["jet_phi"][i])
    E   = float(arrays["jet_energy"][i])
    px = pt * cos(phi); py = pt * sin(phi); pz = pt * sinh(eta)
    m2 = max(E*E - (px*px + py*py + pz*pz), 0.0)
    return jet_sdmass, math.sqrt(m2), pt, eta, phi


# PROCESS ONE CLASS
def process_class(class_name, class_info):
    """Process a single jet class and write its CSV."""
    output_path = os.path.join(OUTPUT_DIR, class_info["output"])
    
    # Skip if exists
    if SKIP_IF_EXISTS and os.path.exists(output_path):
        # Check if file has content
        if os.path.getsize(output_path) > 100:
            print(f"⏭️  Skipping {class_name} - {output_path} already exists")
            return 0
    
    print(f"\n{'='*60}")
    print(f" Processing {class_name}")
    print(f"{'='*60}")
    
    root_files = class_info["files"]
    total_written, wrote_header = 0, False
    target = TARGET_EVENTS_PER_CLASS
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        paths = [f"{os.path.join(ROOT_DIR, fn)}:{TREE_NAME}" for fn in root_files]
        
        try:
            it = uproot.iterate(
                paths,
                expressions=pf_keys,
                entry_step=STEP_SIZE,
                library="np"
            )
        except Exception as e:
            print(f"Error opening files for {class_name}: {e}")
            return 0
        
        for batch_idx, arrays in enumerate(it):
            batch_len = len(arrays["jet_pt"])
            pbar = tqdm(range(batch_len), desc=f"{class_name} batch {batch_idx}", leave=False)
            
            for i in pbar:
                if target is not None and total_written >= target:
                    break
                try:
                    built = build_pf_tensor(arrays, i)
                    if built is None:
                        continue
                    points, features, lorentz_vectors, mask = built
                    
                    with torch.no_grad():
                        out = model(points, features, lorentz_vectors, mask)
                    
                    # Handle both (logits, embedding) and embedding-only cases
                    if isinstance(out, tuple):
                        logits, embedding = out
                    else:
                        logits, embedding = None, out
                    
                    emb = embedding.squeeze(0).detach().cpu().numpy()
                    
                    if not wrote_header:
                        base = ["file","global_index","truth_label","label_name",
                                "jet_sdmass","jet_mass","jet_pt","jet_eta","jet_phi"]
                        emb_cols = [f"emb_{j}" for j in range(emb.shape[-1])]
                        writer.writerow(base + emb_cols)
                        wrote_header = True
                    
                    truth_label, label_name = get_truth_label(arrays, i)
                    jet_sdmass, jet_mass, pt, eta, phi = jet_masses(arrays, i)
                    
                    row = [class_name, total_written, truth_label, label_name,
                           jet_sdmass, jet_mass, pt, eta, phi] + list(emb.astype(np.float32))
                    writer.writerow(row)
                    total_written += 1
                    
                except Exception as e:
                    pbar.set_postfix_str(f"err: {e}")
                    continue
            
            if target is not None and total_written >= target:
                break
    
    print(f"{class_name}: Saved {total_written:,} rows to {output_path}")
    return total_written


# MAIN
def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nJet Class Embedding Generator")
    print(f"  Classes: {len(JET_CLASSES)}")
    print(f"  Events per class: {TARGET_EVENTS_PER_CLASS if TARGET_EVENTS_PER_CLASS else 'ALL'}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Skip existing: {SKIP_IF_EXISTS}\n")
    
    total_events = 0
    for class_name, class_info in JET_CLASSES.items():
        events = process_class(class_name, class_info)
        total_events += events
    
    print(f"\n{'='*60}")
    print(f"DONE! Processed {total_events:,} total events across {len(JET_CLASSES)} classes")
    print(f"Output files in: {OUTPUT_DIR}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
