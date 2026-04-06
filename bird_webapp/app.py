import os
import io
import torch
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
from diffusers import UNet2DModel, DDPMScheduler
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO, emit
from tqdm import tqdm

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HOP_LENGTH = 256
N_MELS = 256
TIME_BINS = 256
FMIN = 20
FMAX = 10000
TOP_DB = 80.0
SR = 24000
N_FFT = 1024

NUM_CLASSES = 5
NULL_CLASS = NUM_CLASSES

label2idx = {
    "American Robin": 0, "Bewick's Wren": 1,
    "Northern Cardinal": 2, "Northern Mockingbird": 3,
    "Song Sparrow": 4,
}
idx2label = {v: k for k, v in label2idx.items()}
idx2label[NULL_CLASS] = "Unlabeled"

print(f"Initializing Diffusion Model Architecture on {DEVICE}...")
model = UNet2DModel(
    sample_size=(N_MELS, TIME_BINS),
    in_channels=1, out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),
    num_class_embeds=NUM_CLASSES + 1,
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(DEVICE)

# Relative to where the command is run. Assuming the user runs `python app.py` from your specified project root inside linux.
# The Notebook saved it to `checkpoints/bird_diffusion_unet_best_ema.pt` but user specified `bird_diffusion_unet_best_30epoch.pt`.
model_path = "../bird_diffusion_unet_best_30epoch.pt"

# Mock the model loading gracefully if run locally without weights in Dev Mode
if os.path.exists(model_path):
    print(f"Loading checkpoint {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
else:
    print(f"WARNING: Checkpoint '{model_path}' not found! The UI will function but will output dummy audio sweeps.")

model.eval()

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="epsilon",
)

def mel_norm_to_mel_db(mel_norm):
    return (mel_norm - 1.0) * (TOP_DB / 2.0)

def mel_norm_to_audio(mel_norm, n_iter=64):
    if isinstance(mel_norm, torch.Tensor):
        mel_norm = mel_norm.detach().cpu().float().numpy()
    if mel_norm.ndim == 3:
        mel_norm = mel_norm.squeeze(0)
    mel_db = mel_norm_to_mel_db(mel_norm)
    mel_power = librosa.db_to_power(mel_db)
    y_hat = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX, power=2.0, n_iter=n_iter)
    return y_hat.astype(np.float32)

@torch.no_grad()
def generate_interpolated(species_weights, num_samples=1, guidance_scale=3.0, device=DEVICE, progress_callback=None):
    """Generate mel spectrograms from a BLEND of multiple bird species."""
    items = []
    total_w = sum(species_weights.values())
    if total_w == 0:
        total_w = 1.0 
        
    for key, w in species_weights.items():
        if w > 0:
            idx = label2idx[key]
            items.append((idx, w / total_w))

    if not items:
         items = [(0, 1.0)]

    emb_list = []
    for idx, w in items:
        emb = model.class_embedding(torch.tensor([idx], device=device).long())
        emb_list.append(emb * w)
    
    blended_emb = sum(emb_list).expand(num_samples, -1)

    null_emb = model.class_embedding(
        torch.tensor([NULL_CLASS], device=device).long()
    ).expand(num_samples, -1)

    original_class_embedding = model.class_embedding

    class BlendedEmbedding(torch.nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.emb = emb
        def forward(self, x):
            return self.emb

    try:
        sample = torch.randn(num_samples, 1, N_MELS, TIME_BINS, device=device)
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        dummy_labels = torch.zeros(num_samples, dtype=torch.long, device=device)

        # Skip actual 1000 steps inference loop if we don't have real weights.
        if os.path.exists(model_path):
            timesteps = scheduler.timesteps
            total_steps = len(timesteps)
            for i, t in enumerate(timesteps):
                if progress_callback:
                    progress_callback(int((i / total_steps) * 100), i, total_steps)
                
                t_batch = t.expand(num_samples).to(device)
                model.class_embedding = BlendedEmbedding(blended_emb)
                noise_cond = model(sample, t_batch, class_labels=dummy_labels).sample
                
                model.class_embedding = BlendedEmbedding(null_emb)
                noise_uncond = model(sample, t_batch, class_labels=dummy_labels).sample
                
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                sample = scheduler.step(noise_pred, t, sample).prev_sample
    finally:
        model.class_embedding = original_class_embedding

    return sample.clamp(-1, 1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    weights = data.get("weights", {})
    
    if os.path.exists(model_path):
        def on_progress(p, current_step, total_steps):
            socketio.emit('progress', {
                'percentage': p,
                'current_step': current_step,
                'total_steps': total_steps
            })
            socketio.sleep(0) # Yield for socket emission
            
        mel = generate_interpolated(weights, num_samples=1, guidance_scale=3.0, progress_callback=on_progress)
        audio = mel_norm_to_audio(mel)
    else:
        # Generate dummy 3s tone mapped to mixed weights if missing checkpoint
        t = np.linspace(0, 3, SR * 3, endpoint=False)
        freq = sum([val * (400 * (i+1)) for i, val in enumerate(weights.values())])
        audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)

    buf = io.BytesIO()
    audio = np.clip(audio, -1, 1)
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(buf, SR, audio_int16)
    buf.seek(0)
    
    return send_file(buf, mimetype="audio/wav")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
