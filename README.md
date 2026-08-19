# Bright ≠ Near

**Auditing and Mitigating Illumination Shortcuts in Ground-Truth-Free Endoscopic Depth**

Rayan Albokhari, Chenyu Zhang, Rui Gao, Zexi Li, Jens Rittscher
University of Oxford

Accepted at the joint AE-CAI | CARE | OR 2.0 | PRiSM workshop at MICCAI 2026, and under
consideration for the Special Issue of Wiley IET Healthcare Technology Letters.

> On real endoscopic tissue there is no metric depth ground truth, so monocular depth is
> ranked by correlation metrics against weak references. We show this is unsafe: a
> zero-parameter "bright = near" baseline is not beaten by the best learned models, because
> a co-located camera and light make brightness a near-field depth cue. We audit the
> confound, show with interventional relighting that the reliance is a shortcut rather than
> a valid cue, release a 307-image 4-annotator human-consensus 5-point ordinal benchmark on
> real colonoscopy with a brightness-discordant protocol, and take a preliminary,
> label-dependent step towards mitigation.

**Project page:** https://albokhari.github.io/bright-not-near/

## Release contents

| Directory | Contents |
|---|---|
| `annotations/` | Point coordinates and ranks, all per-annotator rankings, consensus / non-consensus labels. Keyed to Kvasir-SEG image identifiers (see note below). |
| `folds/` | Image-level fold definitions, plus the cluster-disjoint robustness folds. |
| `code/` | The cue audit, the relighting operators, the evaluation protocol (pairwise / ExactMatch / brightness-discordant), and the fixed-geometry relighting experiment (Exp A). |
| `heads/` | Trained DAV2 depth heads: ordinal fine-tune and relight-consistency variant, per fold. |

**Kvasir-SEG images are not redistributed here.** The Kvasir-SEG terms allow research use
but not redistribution, so every annotation is keyed to the original Kvasir-SEG image
identifier. Download the images from the official Kvasir-SEG source and the loaders in
`code/` will join them to the annotations.

## Status

This repository accompanies the paper and is being populated for the camera-ready release.

## Citation

```bibtex
@article{albokhari2026brightnotnear,
  title   = {Bright $\neq$ Near: Auditing and Mitigating Illumination Shortcuts in
             Ground-Truth-Free Endoscopic Depth},
  author  = {Albokhari, Rayan and Zhang, Chenyu and Gao, Rui and Li, Zexi and Rittscher, Jens},
  journal = {Healthcare Technology Letters},
  year    = {2026},
  note    = {AE-CAI | CARE | OR 2.0 | PRiSM workshop at MICCAI 2026}
}
```

## Acknowledgements

J. Rittscher was funded by the National Institute for Health Research (NIHR) Oxford
Biomedical Research Centre. The views expressed are those of the authors and not
necessarily those of the National Health Service, the NIHR, or the Department of Health.
