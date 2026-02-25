# MIE1517_Project_W2026

Generative cross-breed bird song by bird names.


## 📌 Overview

Briefly describe:

- The objective of this project is to train a neural network capable of generating bird song that simulate high-fidelity cross-breeds of different species.

- We use a VAE UNet architecture with reconstruction and KL divergence loss. The system receives a conditioning input (i.e., bird name) and outputs a synthesized audio waveform representing the generated bird song.

- Key results (if available)


## 🔧 TODO

- Add residual connections to UNet


## 🏗️ Project Structure

```
project-name/
│
├── checkpoints/           # Saved models
├── configs/               # YAML / JSON config files
├── docs/                  # Documentation
├── src/
│   ├── models/            # Model definitions
│   ├── utils/             # Utility functions
│   └── __init__.py
│
├── train.py
└── README.md
```


## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/project-name.git
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
venv\Scripts\activate            # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```


## ▶️ Usage

### Train

```bash
python3 train.py
```

### Evaluate

```bash
python3 evaluate.py --checkpoint checkpoints/model.pt
```

### Inference

```bash
python3 predict.py --input sample.yaml
```