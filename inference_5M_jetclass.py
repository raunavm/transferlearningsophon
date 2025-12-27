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

TARGET_EVENTS = 5_000_000      
MAX_PART = 128                 
STEP_SIZE = 5000               
TREE_NAME = "tree"            

root_dir = "data/JetClass/val_5M"

# Other class file names
# root_files = ["HToBB_100.root", "HToBB_101.root", "HToBB_102.root", "HToBB_103.root", "HToBB_104.root"]  # Hbb
# root_files = ["HToGG_140.root", "HToGG_141.root", "HToGG_142.root", "HToGG_143.root", "HToGG_144.root"]  # Hgg
# root_files = ["HToWW4Q_160.root", "HToWW4Q_161.root", "HToWW4Q_162.root", "HToWW4Q_163.root", "HToWW4Q_164.root"]  # H4q
# root_files = ["HToWW2Q1L_180.root", "HToWW2Q1L_181.root", "HToWW2Q1L_182.root", "HToWW2Q1L_183.root", "HToWW2Q1L_184.root"]  # Hqql
# root_files = ["ZToQQ_200.root", "ZToQQ_201.root", "ZToQQ_202.root", "ZToQQ_203.root", "ZToQQ_204.root"]  # Zqq
# root_files = ["WToQQ_220.root", "WToQQ_221.root", "WToQQ_222.root", "WToQQ_223.root", "WToQQ_224.root"]  # Wqq
# root_files = ["TTBar_240.root", "TTBar_241.root", "TTBar_242.root", "TTBar_243.root", "TTBar_244.root"]  # Tbqq
# root_files = ["TTBarLep_260.root", "TTBarLep_261.root", "TTBarLep_262.root", "TTBarLep_263.root", "TTBarLep_264.root"]  # Tbl
# root_files = ["ZJetsToNuNu_280.root", "ZJetsToNuNu_281.root", "ZJetsToNuNu_282.root", "ZJetsToNuNu_283.root", "ZJetsToNuNu_284.root"]  # Znn
# root_files = ["QCD_000.root", "QCD_001.root", "QCD_002.root", "QCD_003.root", "QCD_004.root"]  # QCD
root_files = [
    "HToCC_120.root", "HToCC_121.root",
    "HToCC_122.root", "HToCC_123.root", "HToCC_124.root"
]
OUTPUT_CSV = "inference_5M_with_embedding.csv"

# Set to True to skip regenerating CSV if it already exists
SKIP_IF_EXISTS = True

particle_keys = [
    'part_px', 'part_py', 'part_pz', 'part_energy',
    'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
    'part_dzval', 'part_dzerr', 'part_charge',
    'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon'
]
scalar_keys = [
    'label_QCD','label_Hbb','label_Hcc','label_Hgg',
    'label_H4q','label_Hqql','label_Zqq','label_Wqq',
    'label_Tbqq','label_Tbl','jet_pt','jet_eta','jet_phi',
    'jet_energy','jet_nparticles','jet_sdmass','jet_tau1',
    'jet_tau2','jet_tau3','jet_tau4','aux_genpart_eta',
    'aux_genpart_phi','aux_genpart_pid','aux_genpart_pt',
    'aux_truth_match'
]
pf_keys = particle_keys + scalar_keys

label_names = ["QCD","Hbb","Hcc","Hgg","Htoww4q","Hqql","Zqq","Znn","Htoww2q1L","Ttbar", "Ttbarlep"]


class DummyDataConfig:
    input_dicts = {"pf_features": list(range(37))}
    input_names = ["pf_points"]
    input_shapes = {"pf_points": (MAX_PART, 37)}
    label_names = ["label"]
    num_classes = 10

data_config = DummyDataConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = get_model(data_config, num_classes=data_config.num_classes, export_embed=True)
model.eval().to(device)

def build_pf_tensor(arrays, i):
    """Return model inputs for event i, or None if too many particles."""
    n_part = arrays["part_px"][i].shape[0]
    if n_part > MAX_PART:
        return None
    particle_feats = [arrays[k][i] for k in particle_keys]
    scalar_feats = [np.full(n_part, arrays[k][i]) for k in scalar_keys]
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
    return y, label_names[y]

def jet_masses(arrays, i):
    jet_sdmass = float(arrays["jet_sdmass"][i])
    pt  = float(arrays["jet_pt"][i])
    eta = float(arrays["jet_eta"][i])
    phi = float(arrays["jet_phi"][i])
    E   = float(arrays["jet_energy"][i])
    px = pt * cos(phi); py = pt * sin(phi); pz = pt * sinh(eta)
    m2 = max(E*E - (px*px + py*py + pz*pz), 0.0)
    return jet_sdmass, math.sqrt(m2), pt, eta, phi


def main():
    # Skip regeneration if CSV already exists
    if SKIP_IF_EXISTS and os.path.exists(OUTPUT_CSV):
        print(f"✅ CSV already exists: {OUTPUT_CSV} (skipping inference, set SKIP_IF_EXISTS=False to regenerate)")
        return
    
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    total_written, wrote_header = 0, False

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        paths = [f"{os.path.join(root_dir, fn)}:{TREE_NAME}" for fn in root_files]
        it = uproot.iterate(
            paths,
            expressions=pf_keys,
            entry_step=STEP_SIZE,
            library="np"  
        )

        for batch_idx, arrays in enumerate(it):
            batch_len = len(arrays["jet_pt"])
            pbar = tqdm(range(batch_len), desc=f"batch {batch_idx}")

            for i in pbar:
                if total_written >= TARGET_EVENTS:
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
                        logits, embedding = None, out  # treat single output as embedding

                    emb = embedding.squeeze(0).detach().cpu().numpy()


                    if not wrote_header:
                        base = ["file","global_index","truth_label","label_name",
                                "jet_sdmass","jet_mass","jet_pt","jet_eta","jet_phi"]
                        emb_cols = [f"emb_{j}" for j in range(emb.shape[-1])]
                        writer.writerow(base + emb_cols)
                        wrote_header = True

                    truth_label, label_name = get_truth_label(arrays, i)
                    jet_sdmass, jet_mass, pt, eta, phi = jet_masses(arrays, i)

                    row = ["unknown", total_written, truth_label, label_name,
                           jet_sdmass, jet_mass, pt, eta, phi] + list(emb.astype(np.float32))
                    writer.writerow(row)
                    total_written += 1

                except Exception as e:
                    pbar.set_postfix_str(f"err: {e}")
                    continue

            if total_written >= TARGET_EVENTS:
                break

    print(f"\n✅ Saved {total_written:,} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
