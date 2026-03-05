"""
V-JEPA based Deepfake Video Detector
Frozen encoder + lightweight binary classification probe
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import io
from typing import Tuple


class AttentiveProbe(nn.Module):
    """
    Lightweight 2-layer attention probe trained on top of frozen V-JEPA features.
    This is the ONLY part you train — encoder stays frozen.
    """
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.norm  = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)       # 2 classes: real / deepfake
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, T*patches, embed_dim]
        B = features.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, features, features)
        out = self.norm(out.squeeze(1))
        return self.classifier(out)


class VJEPADeepfakeDetector:
    """
    Full pipeline: video bytes → V-JEPA features → binary prediction
    """

    def __init__(self, encoder_weights: str, probe_weights: str = None,
                 device: str = "cpu"):
        self.device = device

        # ── Load frozen V-JEPA encoder ──────────────────────────────
        from src.models.vision_transformer import vit_huge   # from jepa repo
        self.encoder = vit_huge(patch_size=16, num_frames=16)
        ckpt = torch.load(encoder_weights, map_location=device)
        self.encoder.load_state_dict(ckpt.get("encoder", ckpt), strict=False)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False         # FROZEN — never updated

        # ── Attentive probe (trainable) ─────────────────────────────
        self.probe = AttentiveProbe(embed_dim=1280)
        if probe_weights:
            self.probe.load_state_dict(torch.load(probe_weights, map_location=device))
        self.probe.to(device)

    # ── Inference ──────────────────────────────────────────────────────
    def predict(self, video_bytes: bytes) -> dict:
        """Run full deepfake detection pipeline on video bytes."""
        frames = self._extract_frames(video_bytes, num_frames=16)
        tensor = self._preprocess(frames)            # [1, 3, T, H, W]

        with torch.no_grad():
            features = self.encoder(tensor)          # [1, N_tokens, 1280]
            logits   = self.probe(features)          # [1, 2]
            probs    = torch.softmax(logits, dim=-1)

        real_prob  = float(probs[0][0])
        fake_prob  = float(probs[0][1])
        verdict    = "DEEPFAKE" if fake_prob > 0.5 else "REAL"
        confidence = round(max(real_prob, fake_prob) * 100, 1)

        return {
            "verdict":          verdict,
            "deepfake_prob":    round(fake_prob * 100, 1),
            "real_prob":        round(real_prob * 100, 1),
            "confidence":       confidence,
            "frames_analyzed":  len(frames),
        }

    # ── Internal helpers ───────────────────────────────────────────────
    def _extract_frames(self, video_bytes: bytes, num_frames: int = 16):
        """Extract evenly spaced frames from video bytes using OpenCV."""
        # Write bytes to temp buffer OpenCV can read
        arr  = np.frombuffer(video_bytes, dtype=np.uint8)
        cap  = cv2.VideoCapture()
        cap.open(cv2.CAP_ANY)

        # Fallback: write to temp file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)

        cap.release()
        os.unlink(tmp_path)
        return frames

    def _preprocess(self, frames: list) -> torch.Tensor:
        """Normalize and stack frames into model input tensor."""
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        processed = []
        for f in frames:
            f = f.astype(np.float32) / 255.0
            f = (f - mean) / std
            processed.append(f)

        arr = np.stack(processed)                    # [T, H, W, 3]
        arr = arr.transpose(3, 0, 1, 2)              # [3, T, H, W]
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

    # ── Training (run once with FaceForensics++ dataset) ───────────────
    def train_probe(self, train_loader, epochs: int = 10, lr: float = 1e-4):
        """
        Train only the attentive probe — encoder stays frozen.
        train_loader yields (video_tensor, label) — label: 0=real, 1=fake
        """
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss, correct = 0, 0
            for videos, labels in train_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    features = self.encoder(videos)   # frozen

                logits = self.probe(features)
                loss   = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct    += (logits.argmax(1) == labels).sum().item()

            acc = correct / len(train_loader.dataset) * 100
            print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}  Acc: {acc:.1f}%")

        torch.save(self.probe.state_dict(), "deepshield_probe.pth")
        print("Probe saved → deepshield_probe.pth")