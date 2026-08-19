# Trained depth heads

Fine-tuned DPT depth heads for Depth Anything V2 (ViT-L). The DINOv2 encoder is frozen in
all our fine-tuning; each checkpoint is the `depth_head` state_dict only (~124 MB), loaded
as:

```python
from depth_anything_v2.dpt import DepthAnythingV2
model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load("depth_anything_v2_vitl.pth"))       # official DAV2 weights
model.depth_head.load_state_dict(torch.load("head_best.pt"))          # our fine-tuned head
```

Checkpoints exceed GitHub's file-size limit, so they are attached to the
**v1.0 GitHub Release** of this repository as:

- `dav2_ordinalFT_heads.tar.gz` — plain ordinal fine-tune, folds 0-4
- `dav2_relightcons_heads.tar.gz` — ordinal + relight-consistency (the cure), folds 0-4

The `dav2_ordinalFT_fold*/` and `dav2_relightcons_fold*/` directories here hold each run's
`summary.json` (held-out metrics) for reference. Training recipes: `../code/finetune/`.
