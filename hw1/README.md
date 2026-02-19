# AIAA5032 HW1 — Audio-Based Video Classification

**Best Kaggle Score: 0.72 (1st place)**
Approach: BiLSTM + 5-Fold Cross-Validation Ensemble + Test-Time Augmentation (TTA)

---

## Directory Structure

```
hw1/
├── run_bilstm_kfold.py   ← MAIN script (best result)
├── run_bilstm.py         ← Single BiLSTM (single-split reference)
├── infer_bilstm.py       ← Run inference from a saved checkpoint
├── split_train_val.py    ← Split trainval.csv → train.csv + val.csv
├── select_frames.py      ← MFCC frame sampling utility (from baseline)
├── labels/               ← trainval.csv, train.csv, val.csv, test_for_student.label
├── experiments/          ← Saved experiment outputs
└── old_scripts/          ← Earlier attempts (for reference)
    ├── baseline/         ← BoF + SVM / LR / MLP baselines
    ├── mlp/              ← MLP hyperparameter search, PCA-MLP, Bagging-MLP
    ├── fisher_vector/    ← GMM + Fisher Vector pipeline
    ├── xgboost/          ← XGBoost experiments
    ├── cnn/              ← 1D-CNN experiment
    └── ensemble/         ← Post-hoc ensemble scripts
```

---

## Environment

```bash
conda create -n hw1 python=3.11
conda activate hw1
pip install torch==2.0.1 scikit-learn numpy
```

> Tested on Ubuntu 22.04, CUDA 11.8. CPU-only also works (slower).

---

## How to Reproduce the Best Result

### 1. Prepare labels

```bash
mkdir -p labels/
# Place trainval.csv and test_for_student.label under labels/
python split_train_val.py   # generates labels/train.csv and labels/val.csv
```

### 2. Unzip MFCC features

```bash
tar zxvf mfcc.tgz
```

### 3. Run BiLSTM K-Fold training (produces test predictions)

```bash
python run_bilstm_kfold.py \
    /path/to/mfcc/ \
    labels/trainval.csv \
    labels/test_for_student.label \
    --epochs 150 --patience 20 \
    --n_folds 5 --n_tta 8 \
    --label_smoothing 0.1
```

Output: `experiments/bilstm_kfold_<timestamp>/test_kfold_tta.csv`

Upload that CSV to Kaggle.

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--hidden_dim` | 256 | BiLSTM hidden size per direction |
| `--num_layers` | 2 | Number of LSTM layers |
| `--dropout` | 0.5 | Dropout rate |
| `--lr` | 1e-3 | Initial learning rate |
| `--epochs` | 150 | Max epochs per fold (early stopping applies) |
| `--patience` | 20 | Early stopping patience |
| `--n_folds` | 5 | Number of CV folds |
| `--n_tta` | 8 | TTA passes at inference |
| `--label_smoothing` | 0.1 | Label smoothing coefficient |

---

## Method

### Model: Bidirectional LSTM with Attention Pooling

```
Input MFCC (T × 39)
    → BatchNorm
    → BiLSTM (2 layers, hidden=256, bidirectional)
    → Attention Pooling (learnable scalar weights over time)
    → FC(512 → 256) → ReLU → Dropout
    → FC(256 → 10)
```

Variable-length sequences are handled via PyTorch `pack_padded_sequence`.

### Data Augmentation (training only)

Applied per-sample at every training step:

- **Speed perturbation**: randomly stretch/compress time axis by ×0.8–1.2
- **Time masking**: 2 × up to 10% of frames zeroed out
- **Frequency masking**: 2 × up to 15% of MFCC bins zeroed out
- **Gaussian noise**: σ = 0.05 added to all frames
- **CMVN**: cepstral mean-variance normalisation per utterance

### Training Strategies

| Strategy | Details |
|---|---|
| **5-Fold Stratified CV** | Each fold: 80% train / 20% val; 5 independent models |
| **Label Smoothing** | ε = 0.1; prevents overconfident predictions |
| **Cosine LR Schedule** | LR decays from 1e-3 → 1e-5 over `--epochs` steps |
| **Gradient Clipping** | max norm = 5.0 |
| **Early Stopping** | patience = 20 on fold val accuracy |
| **TTA at inference** | Each test sample augmented 8 times; softmax averaged |
| **Ensemble** | Final prediction = average of 5 × 8 = 40 soft predictions |

---

## Results Summary

| Method | Val Acc | Kaggle Score |
|---|---|---|
| BoF (k=50) + SVM | ~55% | — |
| BoF (k=50) + MLP | ~58% | — |
| Fisher Vector (k=200) + MLP | ~67% | 0.629 |
| Fisher Vector (k=300) + MLP | ~69% | — |
| BiLSTM (single model, train split) | 69.4% | 0.66 |
| **BiLSTM K-Fold + TTA + Label Smoothing** | **68.6% (CV mean)** | **0.72 (1st)** |

---

## Experiment Journey (see `old_scripts/`)

The dataset has ~5,000 training samples across 10 classes (~500 per class), which makes
overfitting the central challenge throughout all experiments.

---

### Stage 1 — Baselines: BoF + Classical Classifiers (`old_scripts/baseline/`)

**Approach:** Follow the provided pipeline — K-Means codebook (k=50) on MFCC frames →
Bag-of-Features (BoF) histogram per video → train SVM / LR / MLP.

**Results:**

| Classifier | Val Acc |
|---|---|
| SVM (RBF) | ~55% |
| Logistic Regression | ~54% |
| MLP (200, 100) | ~58% |

**Observation:** BoF destroys temporal ordering and quantises features coarsely.
MLP was the best classical classifier, so all subsequent classical experiments kept MLP as the
classifier head.

---

### Stage 2 — Better Features: Fisher Vector (`old_scripts/fisher_vector/`)

**Motivation:** BoF uses hard assignment (each frame goes to exactly one codeword).
Fisher Vector (FV) uses a GMM soft assignment and encodes both first- and second-order
statistics, producing a richer fixed-length video representation.

**Steps:**
1. Train a GMM with k=200 components on a random subset of MFCC frames (`run_gmm.py`)
2. Encode each video as a Fisher Vector of dimension 2 × 200 × 39 = 15,600 (`run_fisher_vector.py`)
3. Train MLP / LR / SVM on the FV features

**Results:**

| Features | Classifier | Val Acc | Kaggle |
|---|---|---|---|
| Fisher Vector k=200 | MLP | 67.65% | 0.629 |
| Fisher Vector k=300 | MLP | **69.18%** | — |

Increasing the GMM components from 200 to 300 (feature dim: 23,400) improved accuracy by
~1.5 pp. This became the strongest classical baseline.

---

### Stage 3 — Pushing the Classical Pipeline (`old_scripts/mlp/`, `old_scripts/xgboost/`)

Three directions were explored to squeeze more out of the Fisher Vector features:

**3a. XGBoost** (`old_scripts/xgboost/`): Replaced MLP with XGBoost on Fisher Vector k=200.
Result: val 66.24% — *worse* than MLP. High-dimensional Fisher Vectors (15,600-dim) do not
suit tree-based models, which cannot exploit the dense linear structure that MLPs handle well.

**3b. MLP architecture search** (`old_scripts/mlp/run_mlp_search.py`): Grid search over
hidden sizes, learning rates, and regularisation strengths. Best config: (512, 256) hidden
units, val 68.12%. Marginal gain over the default (200, 100).

**3c. PCA + MLP** (`old_scripts/mlp/run_pca_mlp.py`): Reduced Fisher Vector to 256 PCA
components before MLP. Result: val 65.41% — *worse*. PCA retained only ~60–70% of variance,
discarding useful discriminative information.

**3d. Bagging MLP** (`old_scripts/mlp/run_bagging_mlp.py`): Trained 10 MLPs on different
random splits of trainval data, averaged softmax probabilities. Result: val 67.18% — no
improvement over a single well-tuned MLP. The bottleneck was the feature quality, not
classifier variance.

**Key insight from Stage 3:** The classical pipeline had plateaued at ~69%. Richer features
or a model that could exploit temporal structure were needed.

---

### Stage 4 — Sequence Modelling: BiLSTM (`run_bilstm.py`)

**Motivation:** All classical methods aggregate MFCC frames into a single fixed-length vector,
losing temporal dynamics. A recurrent model can operate directly on the raw variable-length
MFCC sequence.

**Architecture:**
- 2-layer Bidirectional LSTM (hidden=256 per direction)
- Attention Pooling over all time steps (instead of last hidden state)
- Batch Normalisation on input, Dropout (p=0.5) in the classifier head

**Overfitting problem:** Training accuracy quickly reached 85–90% while validation
accuracy plateaued at 65–68%, with a 20+ pp gap. Several fixes were tried:

| Attempt | Val Acc | Note |
|---|---|---|
| Base BiLSTM (hidden=256, dropout=0.5) | 68% | Strong overfitting |
| Smaller model (hidden=128, dropout=0.7) | 66.6% | Less capacity hurt accuracy too |
| + Speed perturbation (×0.8–1.2) | +1–2 pp | Most effective single augmentation |
| + Time & frequency masking (SpecAugment-style) | +0.5–1 pp | Complementary to speed perturb |
| + Gaussian noise + CMVN | marginal | Stabilised training |
| **Full augmentation stack** | **69.41%** | Best single-split result |

**Observation:** Data augmentation was the key lever. Without it, the BiLSTM overfit
badly; with the full stack it matched and slightly exceeded the Fisher Vector MLP.

---

### Stage 5 — 1D-CNN (`old_scripts/cnn/`)

**Motivation:** Test whether convolutional inductive bias (local pattern detection) would
help compared to recurrent models.

**Architecture:** Residual 1D-CNN blocks with stride-based downsampling (total stride = 8),
masked global average pooling over variable-length sequences. Same augmentation as BiLSTM.

**Result:** Val 63.29%, with highly unstable validation loss oscillating between 1.3 and 3.9.
The CNN was significantly harder to train on this small dataset and was abandoned.

---

### Stage 6 — Final: K-Fold Ensemble + TTA + Label Smoothing (`run_bilstm_kfold.py`)

**Motivation:** The single-split BiLSTM reached 69.41% but its Kaggle score was 0.66,
suggesting the single validation split introduced variance. Three complementary techniques
were combined to address this:

**1. 5-Fold Stratified Cross-Validation:**
Instead of one train/val split, train 5 independent BiLSTM models, each on a different 80%
of trainval data. This ensures every sample is used for training in 4 out of 5 folds, and
the final test prediction is an average of 5 diverse models.

**2. Test-Time Augmentation (TTA):**
At inference, each test sample is augmented 8 times with the same random augmentation used
during training (speed perturbation, masking, noise). The 8 softmax outputs are averaged,
reducing prediction variance from stochastic augmentation.

**3. Label Smoothing (ε = 0.1):**
Replaces hard one-hot targets (1.0 / 0.0) with soft targets (0.9 / 0.011), preventing the
model from becoming overconfident on training labels and improving calibration.

**Final ensemble:** 5 models × 8 TTA passes = **40 soft predictions averaged** per test sample.

**Results:**

| Fold | Val Acc |
|---|---|
| Fold 1 | 67.58% |
| Fold 2 | 67.31% |
| Fold 3 | 68.11% |
| Fold 4 | 68.90% |
| Fold 5 | 70.94% |
| **Mean** | **68.57%** |
| **Kaggle** | **0.72 (1st place)** |

The ~3.4 pp gain over the single-model Kaggle score (0.66 → 0.72) comes from the ensemble
reducing both bias (more training data per model on average) and variance (diversity across
folds and TTA passes).
