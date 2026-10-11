"""
cnn_lstm.py
-----------
Architecture: per-frame spatial features from an EfficientNet-B4 backbone
(pretrained on ImageNet via timm) are fed, frame by frame, into an LSTM
that models temporal artifacts across the sequence (e.g. unnatural blink
patterns, flicker, inconsistent lighting between frames). The LSTM's
final hidden state is concatenated with handcrafted blink-pattern
features and passed through a small classification head.
"""
import torch
import torch.nn as nn
import timm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class EfficientNetFeatureExtractor(nn.Module):
    """Wraps a timm EfficientNet-B4 and strips the classification head,
    leaving a pooled feature vector per frame."""

    def __init__(self, backbone_name=config.CNN_BACKBONE, pretrained=True,
                 freeze=config.FREEZE_CNN_BACKBONE):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0  # num_classes=0 -> pooled features only
        )
        self.feature_dim = self.backbone.num_features

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, feature_dim)
        return self.backbone(x)


class DeepFakeCNNLSTM(nn.Module):
    """
    Forward pass:
      frames: (B, T, 3, H, W)  - a sequence of T face crops per video
      blink_features: (B, 5)   - handcrafted EAR/blink summary features (optional)

    Pipeline:
      1. Flatten (B, T) -> (B*T) and run every frame through the shared CNN backbone.
      2. Reshape back to (B, T, feature_dim) and feed through an LSTM.
      3. Take the LSTM's final hidden state, concatenate blink features.
      4. Classify real (0) vs fake (1).
    """

    def __init__(
        self,
        cnn_feature_dim=config.CNN_FEATURE_DIM,
        lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        lstm_dropout=config.LSTM_DROPOUT,
        num_classes=config.NUM_CLASSES,
        use_blink_features=True,
        blink_feature_dim=5,
    ):
        super().__init__()
        self.cnn = EfficientNetFeatureExtractor()
        self.use_blink_features = use_blink_features

        self.lstm = nn.LSTM(
            input_size=self.cnn.feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=True,
        )

        head_input_dim = lstm_hidden_dim * 2  # bidirectional
        if use_blink_features:
            head_input_dim += blink_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, frames, blink_features=None):
        B, T, C, H, W = frames.shape

        # Run CNN on every frame in the batch at once
        frames_flat = frames.view(B * T, C, H, W)
        cnn_feats = self.cnn(frames_flat)                    # (B*T, feature_dim)
        cnn_feats = cnn_feats.view(B, T, -1)                  # (B, T, feature_dim)

        lstm_out, (h_n, c_n) = self.lstm(cnn_feats)
        # h_n: (num_layers * num_directions, B, hidden_dim) -> take last layer, both directions
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        temporal_feat = torch.cat([forward_hidden, backward_hidden], dim=1)  # (B, hidden_dim*2)

        if self.use_blink_features and blink_features is not None:
            combined = torch.cat([temporal_feat, blink_features], dim=1)
        else:
            combined = temporal_feat

        logits = self.classifier(combined)
        return logits

    def unfreeze_backbone(self, num_last_blocks=None):
        """Call this after a few epochs of head-only training to fine-tune the CNN.
        If num_last_blocks is given, only unfreeze that many trailing blocks."""
        params = list(self.cnn.backbone.parameters())
        if num_last_blocks is None:
            for p in params:
                p.requires_grad = True
        else:
            for p in params[-num_last_blocks:]:
                p.requires_grad = True


def build_model():
    model = DeepFakeCNNLSTM()
    return model.to(config.DEVICE)
