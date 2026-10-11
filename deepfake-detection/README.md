# DeepFake Detection System

A web-based deepfake detector combining spatial features from an
**EfficientNet-B4** CNN with **temporal modeling via a Bi-LSTM**, fused with
handcrafted **blink-pattern (EAR)** features, and served through a **Flask**
app.

```
Video → OpenCV frame sampling → MediaPipe face crop + eye landmarks
      → EfficientNet-B4 (per-frame features) → Bi-LSTM (temporal)
      → concat blink features → classifier → real / fake
```

## Project structure

```
deepfake-detection/
├── config.py                      # all paths & hyperparameters
├── requirements.txt
├── data/
│   ├── raw_videos/{real,fake}/    # <- put downloaded/organized videos here
│   ├── processed/                 # cached .npz face sequences (auto-generated)
│   └── splits/                    # train/val/test.csv (auto-generated)
├── src/
│   ├── preprocessing/
│   │   ├── face_extractor.py      # OpenCV + MediaPipe face crop & EAR
│   │   ├── blink_detector.py      # EAR sequence -> blink feature vector
│   │   └── preprocess_dataset.py  # batch-processes raw_videos/ -> processed/
│   ├── models/cnn_lstm.py         # EfficientNet-B4 + Bi-LSTM architecture
│   ├── dataset.py                 # PyTorch Dataset/DataLoader
│   ├── train.py                   # training loop with early stopping
│   └── evaluate.py                # accuracy / F1 / ROC-AUC on test set
├── app/
│   ├── app.py                     # Flask server
│   ├── inference.py               # single-video prediction pipeline
│   ├── templates/index.html
│   └── static/{style.css, script.js}
└── checkpoints/best_model.pt      # saved after training
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for training (EfficientNet-B4 + LSTM over
20-frame sequences is heavy on CPU). Inference in the Flask app works fine on
CPU, just slower per video.

## 2. Getting the datasets

You don't need both datasets to get started — either one works, and you can
train on a small subset first to verify the pipeline before scaling up.

### Celeb-DF (v2)
- Request access via the official form linked from the authors' GitHub repo:
  **github.com/yuezunli/celeb-deepfakeforensics**. Access is granted for
  research/educational use after you fill out the request form (typically a
  quick approval).
- Once you have the download link, you'll get `Celeb-real/`, `Celeb-synthesis/`
  (fake), and `YouTube-real/` video folders.

### DFDC (DeepFake Detection Challenge)
- The full dataset is very large (~470GB). For a first pass, use the much
  smaller **DFDC sample set** on Kaggle: search "Deepfake Detection Challenge"
  on **kaggle.com/competitions** — the sample zip is a few GB and includes a
  `metadata.json` labeling each clip `REAL`/`FAKE`.
- Kaggle requires a free account; datasets can be pulled with the `kaggle`
  CLI (`pip install kaggle`, then `kaggle competitions download -c
  deepfake-detection-challenge`) once you've accepted the competition rules
  on the site.

### Arranging videos for this project
Whichever dataset(s) you use, sort videos into two folders so the
preprocessing script can find them:

```
data/raw_videos/
├── real/
│   ├── video_0001.mp4
│   └── ...
└── fake/
    ├── video_0001.mp4
    └── ...
```

For DFDC, write a small script to read `metadata.json` and copy/symlink each
clip into `real/` or `fake/` based on its label — happy to help you write
that once you've downloaded a sample if you'd like.

### No dataset yet? Test the pipeline anyway
Drop a handful of any face-containing clips (even just your own short
recordings) into `data/raw_videos/real/` so you can run preprocessing,
training, and the Flask app end-to-end and confirm everything wires up
correctly before you invest time in the full datasets.

## 3. Preprocess videos into face sequences

```bash
python -m src.preprocessing.preprocess_dataset
```

This detects the face in each video, samples a fixed-length sequence of
crops, computes the eye-aspect-ratio (EAR) per frame, and caches everything
as `.npz` files under `data/processed/`. It also writes
`data/splits/{train,val,test}.csv` (70/15/30 split... actually 70/15/15).

## 4. Train

```bash
python -m src.train
```

- Trains with the CNN backbone frozen at first (fast), then unfreezes the
  last few EfficientNet blocks halfway through for fine-tuning at a lower
  learning rate.
- Saves the best checkpoint (by validation loss) to
  `checkpoints/best_model.pt`.
- Edit hyperparameters (batch size, epochs, sequence length, LSTM size) in
  `config.py`.

## 5. Evaluate

```bash
python -m src.evaluate
```

Reports accuracy, F1-score, ROC-AUC, and a confusion matrix on the held-out
test split.

## 6. Run the web app

```bash
python app/app.py
```

Open **http://localhost:5000**, upload a clip, and get a real/fake verdict
with confidence, blink count, and frames analyzed. If no checkpoint exists
yet at `checkpoints/best_model.pt`, the app still runs but returns a clear
message telling you to train a model first.

## Notes on the approach

- **Why blink patterns?** Early face-swap and reenactment models often
  under- or over-produce blinks compared to natural human blink rates
  (~15-20/min), so blink dynamics are a cheap, interpretable signal
  alongside the learned CNN features — this is the same intuition behind
  the "Ictu Oculi" line of deepfake-detection research.
- **Why EfficientNet-B4 + LSTM instead of a 3D CNN?** Treating spatial and
  temporal modeling separately keeps the model lighter to train and lets you
  swap in a stronger/lighter backbone (`timm` supports hundreds) without
  touching the temporal half.
- **Limitations to be upfront about:** this architecture generalizes best to
  the manipulation types present in its training data. A model trained only
  on Celeb-DF will likely underperform on DFDC-style compression artifacts
  and vice versa — training on both, and testing cross-dataset, gives a much
  more honest sense of real-world performance.
