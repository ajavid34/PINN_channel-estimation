# Physics-Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals

Official implementation of **"Physics-Informed Neural Networks for Wireless
Channel Estimation with Limited Pilot Signals"**, Seyed Alireza Javid and Nuria
González-Prelcic (UC San Diego), NeurIPS 2025 Workshop *AI and ML for
Next-Generation Wireless Communications and Networking (AI4NextG)*.

- Paper (OpenReview): <https://openreview.net/pdf?id=r3plaU6DvW>
- Paper (in-repo copy): [`62_Physics_Informed_Neural_Net.pdf`](./62_Physics_Informed_Neural_Net.pdf)


## 📋 Abstract

Accurate wireless channel estimation is critical for next-generation communication systems, yet traditional model-based methods struggle in complex environments with limited pilot signals, while purely data-driven approaches lack physical interpretability and require extensive pilot overhead. This paper presents a novel PINN framework that synergistically combines model-based channel estimation with environmental propagation characteristics to achieve superior performance under pilot-constrained scenarios. The proposed approach employs an enhanced U-Net architecture with transformer modules and cross-attention mechanisms to fuse initial channel estimates with RSS maps. Comprehensive evaluation using realistic ray-tracing data from urban environments demonstrates significant performance improvements, achieving over 5 dB gain in NMSE compared to state-of-the-art methods, with particularly strong performance in pilot-limited scenarios (achieving around -13 dB NMSE with only four pilots at SNR = 0 dB). The proposed framework maintains practical computational complexity, making it viable for massive MIMO systems in upper-mid band frequencies. Unlike black-box neural approaches, the physics-informed design provides a more interpretable channel estimation method.


## Key results

Refined LS-OFDM NMSE (dB) vs. number of pilots `Np` at SNR = 0 dB on the
Boston 15 GHz dataset (Table 2 of the paper):

| `Np`              |   512  |   256  |   128  |    64  |    32  |    16  |     8  |     4  |     2  |
|-------------------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Initial LS-OFDM   | −14.40 | −11.23 |  −8.25 |  −5.49 |  −3.32 |  −1.33 |   0.28 |   1.98 |   3.01 |
| **Refined (PINN)**| **−19.21** | **−17.40** | **−16.20** | **−16.00** | **−13.87** | **−13.85** | **−12.87** | **−12.89** | **−12.35** |
| Improvement (dB)  |  4.81  |  6.17  |  7.95  | 10.51  | 10.55  | 12.52  | 13.15  | 14.87  | 15.36  |

At `Np = 4`, PINN also beats CNN [5], Diffusion [8], OMP, SOMP, and Subspace
Pursuit by roughly 5 dB across the low-SNR regime (Fig. 3 of the paper).


## Repository structure

```
PINN_channel-estimation/
├── Model.py                 # U-Net + transformer + cross-attention + PhysicsInformedLoss
├── find_in_map.py           # RSSMapProcessor: real-world coords <-> pixel, crop around user
├── make_correct_channels.py # Ray-tracing CSV  ->  ground-truth channel tensor (.npy)
├── init_estimation.py       # LS / LS-OFDM initial channel estimators
├── train.py                 # Training entry point for the PINN
├── fine_tune.py             # TransferLearningExperiment (Appendix G)
├── requirements.txt
├── Dataset/                 # Wireless Insite ray-tracing outputs + RSS maps
│   ├── 15GHz_concatenated_data.csv
│   ├── 15GHz_concatenated_data_canyon.csv
│   ├── 8GHz_concatenated_data.csv
│   ├── 50_15GHz.jpg
│   ├── 50_15GHz__canyon.jpg
│   └── 50_8GHz.jpg
└── 62_Physics_Informed_Neural_Net.pdf
```


## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `scipy`, `pandas`, `opencv-python`, `Pillow`,
`matplotlib`, `tqdm`. All experiments in the paper were run on a single
**NVIDIA RTX 4090**; a single full training run (500 epochs) takes on the
order of a few hours on that GPU.


## Dataset

We use ray-traced channels from [Remcom Wireless Insite](https://www.remcom.com/wireless-insite-em-propagation-software), which explicitly models propagation including reflection, diffraction, and scattering in a 3D urban environment. Two environments and two carrier frequencies are provided (Table 5 of the paper):

| Parameter              | Value                    |
|------------------------|--------------------------|
| Environments           | Boston, urban canyon     |
| Map scale              | 350 × 450 m²             |
| Cropped RSS map (per UE) | 10 × 10 m (30 × 30 px) |
| Carrier frequencies    | 15 GHz and 8 GHz         |
| Bandwidth              | 400 MHz (15 GHz), 200 MHz (8 GHz) |
| Tx array `Nt`          | 24 × 24 UPA              |
| Rx array `Nr`          | 2 × 2 UPA                |
| Delay taps `D`         | 16                       |
| Pulse shaping          | Raised cosine, β = 0.4   |
| Transmit power `Pt`    | 50 dBm                   |
| Snapshots              | 9877                     |

`Dataset/` contains:

- **Ray-tracing CSVs** — one row per snapshot, columns `AOD_PHI`, `AOD_THETA`,
  `AOA_PHI`, `AOA_THETA`, `Pathgain`, `ToA`, `PHASE`. Up to 25 MPCs per
  snapshot, exactly as exported from Wireless Insite.
- **RSS map images** — `50_<freq>.jpg` are pre-computed received-power maps
  for `Pt = 50 dBm` over the whole environment, also from Wireless Insite.


### Prerequisites you must supply yourself

These files are *not* generated by the code and must be placed in the repo root
before the regeneration pipeline will run end-to-end:

- **`ue_positions_noisy.txt`** — receiver positions, one row per snapshot (in
  the same order as the ray-tracing CSV). Format: **1 header line + 3 columns
  `x y z`** (whitespace-separated, meters). These come from the UE/receiver
  locations used in your Wireless Insite simulation, with horizontal Gaussian
  noise `N(0, 9·I)` added per the paper's GPS-error model (see Sec. E of the
  paper).

- **Base-station reference coordinates for each scene.** `train.py` already
  has the Boston values hard-coded in its `config` dict:

  ```python
  'bs_pixel_coords':     (287, 293),       # BS location in the RSS jpeg
  'bs_real_coords':      (71.06, 246.29),  # BS location in meters
  'image_width_meters':  527.5,            # horizontal extent of the jpeg
  ```

  If you run experiments on the urban canyon scene, replace these three
  values with the canyon equivalents before training.


## Regenerating the results

The pipeline has four stages. All commands assume you're in the repo root.

### Step 1 — Build ground-truth channel tensors from the ray-tracing CSVs

`make_correct_channels.py` parses the Wireless Insite export and applies the
array-response and raised-cosine sum in Eq. (2) of the paper to produce a
`(num_snapshots, D, Nr, Nt)` complex tensor.

```bash
# Boston, 15 GHz, 400 MHz
python make_correct_channels.py \
    --csv Dataset/15GHz_concatenated_data.csv \
    --out 3D_channel_15GHz_2x2_Pt50.npy \
    --pt 50 --bw 4e8

# Boston, 8 GHz, 200 MHz
python make_correct_channels.py \
    --csv Dataset/8GHz_concatenated_data.csv \
    --out 3D_channel_8GHz_2x2_Pt50.npy \
    --pt 50 --bw 2e8

# Urban canyon, 15 GHz, 400 MHz
python make_correct_channels.py \
    --csv Dataset/15GHz_concatenated_data_canyon.csv \
    --out 3D_channel_15GHz_2x2_Pt50_canyon.npy \
    --pt 50 --bw 4e8
```

Run `python make_correct_channels.py --help` for the full list of flags
(including `--n-tx-x`, `--n-rx-x`, `--n-tap`, `--mask-low-gains`, …).

### Step 2 — Generate the initial LS-OFDM channel estimates

For a given `(SNR, Np)` operating point, `init_estimation.py` simulates
OFDM pilot transmission at `N_subcarriers / pilot_spacing` subcarriers,
performs LS interpolation, and saves a `.npy` with the same shape as Step 1.

```bash
# Example: SNR = 0 dB, Np = 256 (pilot_spacing = 4)
python init_estimation.py \
    --true-channels 3D_channel_15GHz_2x2_Pt50.npy \
    --output initial_estimate_ls_snr0.npy \
    --snr 0 \
    --n-subcarriers 1024 \
    --pilot-spacing 4
```

The relationship used in the paper is `Np = N_subcarriers / pilot_spacing`.
To reproduce Table 2 at SNR = 0 dB you would run this command nine times, one
per column:

| `Np` |  512 |  256 |  128 |  64 |  32 |  16 |   8 |   4 |   2 |
|------|:----:|:----:|:----:|:---:|:---:|:---:|:---:|:---:|:---:|
| `--pilot-spacing` | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |

### Step 3 — Train the PINN

Edit the `config` dict at the bottom of [`train.py`](./train.py) so that
`smomp_file` points at the initial estimate from Step 2 and `accurate_file`
points at the ground-truth tensor from Step 1. Then:

```bash
python train.py
```

This trains for 500 epochs with the hyperparameters from Table 1 of the paper
(see below), saves the best-validation-NMSE checkpoint to `name_val`, and the
last-epoch checkpoint to `name_train`. The script prints the final test-set
NMSE in both linear and dB scale.

### Step 4 — Reproduce Fig. 2 / Fig. 3 / Table 2

Each point in Fig. 2 / Fig. 3 / Table 2 is one training run with a different
`(SNR, Np)` initial estimate. A minimal bash sweep looks like:

```bash
# Reproduce the Table 2 row at SNR = 0 dB
for ps in 2 4 8 16 32 64 128 256 512; do
    Np=$((1024 / ps))
    python init_estimation.py \
        --true-channels 3D_channel_15GHz_2x2_Pt50.npy \
        --output "initial_estimate_snr0_np${Np}.npy" \
        --snr 0 --pilot-spacing $ps

    # Update train.py config to point at the new smomp_file, then:
    python train.py | tee "log_snr0_np${Np}.txt"
done
```

For Fig. 2 (NMSE vs. SNR), loop SNR over `-10, -5, 0, 5, 10, 15, 20` with a
fixed `Np = 64` (`pilot_spacing = 16`). For Fig. 3a, fix `Np = 4`
(`pilot_spacing = 256`) and sweep SNR over the same range.


## Training hyperparameters

Reproduced from Table 1 of the paper and implemented in
[`Model.py:train_model`](./Model.py):

| Hyperparameter | Value |
|----------------|-------|
| Batch size     | 32    |
| Epochs         | 500   |
| Init. LR       | 1 × 10⁻³ |
| Scheduler      | StepLR |
| Decay step     | 40    |
| γ (decay)      | 0.65  |
| Optimizer      | Adam  |
| Momentum       | 0.9   |
| ζ (physics weight) | 0.01 |

Data split: 80% train / 10% validation / 10% test, global normalization by
`max(|H_init|, |H_true|)` applied *before* splitting (seed `42`). The loss is

```
L_total  =  L_NMSE  +  ζ · L_physical
```

where `L_NMSE = E[‖H − Ĥ‖² / ‖H‖²]` is the reconstruction term and
`L_physical` ties predicted channel power per tap to the RSS map reading at
the user's (noisy) position — see Eqs. (6)–(8) in the paper and
`PhysicsInformedLoss` in [`Model.py`](./Model.py).


## Transfer-learning experiments (Appendix G)

`fine_tune.py` implements the full `TransferLearningExperiment` used in the
paper. It loads a checkpoint trained on the "source" scene/frequency and
fine-tunes it on a configurable fraction of a "target" dataset.

### 15 GHz → 8 GHz (Boston, Figure 8a)

1. Run Steps 1–3 above for the 15 GHz Boston dataset. This produces
   `simple_ls_0_val.pth` (best-val checkpoint).
2. Also run Steps 1–2 for the 8 GHz Boston dataset (use `--bw 2e8`).
3. Create a small driver script that instantiates the experiment:

```python
from fine_tune import TransferLearningExperiment

config_15 = {  # source
    'smomp_file':         'initial_estimate_ls_snr0.npy',
    'accurate_file':      '3D_channel_15GHz_2x2_Pt50.npy',
    'user_positions_file':'ue_positions_noisy.txt',
    'rss_image_path':     'Dataset/50_15GHz.jpg',
    'bs_pixel_coords':    (287, 293),
    'bs_real_coords':     (71.06, 246.29),
    'image_width_meters': 527.5,
}

config_8 = {   # target
    'smomp_file':         'initial_estimate_ls_snr0_8GHz.npy',
    'accurate_file':      '3D_channel_8GHz_2x2_Pt50.npy',
    'user_positions_file':'ue_positions_noisy.txt',
    'rss_image_path':     'Dataset/50_8GHz.jpg',
    'bs_pixel_coords':    (287, 293),
    'bs_real_coords':     (71.06, 246.29),
    'image_width_meters': 527.5,
}

exp = TransferLearningExperiment('simple_ls_0_val.pth', config_15, config_8)

# Reproduce Fig. 8a: NMSE vs. fraction of 8 GHz training data.
# sample_sizes[i] ≈ fraction_i × len(train_dataset_8ghz)
results = exp.run_experiment(
    sample_sizes=[50, 100, 200, 500, 1000, 2000, 4000, len(exp.train_dataset_8ghz)],
    epochs_per_size=20,   # Fig. 8a has two curves: 20 and 100 epochs
)
```

Run it once with `epochs_per_size=20` and once with `epochs_per_size=100` to
get both curves in Fig. 8a.

### Boston → urban canyon (Figure 8b)

Same pattern, but:

- The "target" side uses the canyon CSV
  (`Dataset/15GHz_concatenated_data_canyon.csv`) and canyon RSS jpeg
  (`Dataset/50_15GHz__canyon.jpg`).
- **Update `bs_pixel_coords` / `bs_real_coords` / `image_width_meters` to
  match the canyon map** — the Boston values will not work.
- Sweep `sample_sizes = [50, 100, 250, 500, 1000, 1500, 2000]` and again run
  twice at 20 and 100 epochs to match Fig. 8b.


## Computational budget

From Table 4 of the paper, at `(Nt, Nr, D) = (576, 4, 16)`:

| Model | FLOPs | Inference latency | Parameters |
|-------|-------|--------------------|------------|
| **PINN (ours)** | 70.85 G | **11.12 ms** | 3.5 × 10⁸ |
| Diffusion baseline | 130.15 G | 50.30 ms | 5.5 × 10⁴ |

All measurements taken on a single NVIDIA RTX 4090. One full training run
(500 epochs, batch 32, 9877 × 0.8 ≈ 7900 train samples) takes on the order of
a few hours on that GPU.


## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{javid2025physics,
  title={Physics-Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals},
  author={Javid, Alireza and Prelcic, Nuria Gonzalez},
  booktitle={NeurIPS 2025 Workshop: AI and ML for Next-Generation Wireless Communications and Networking}
}
```
