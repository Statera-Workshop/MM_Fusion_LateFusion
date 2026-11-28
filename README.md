<!-- PAI-SG-LINKS:BEGIN -->
## Project Links

- [Project Portal](https://statera-workshop.github.io/#repo-mm_fusion_latefusion)
- [Statera Workshop Portal](https://statera-workshop.github.io/)
<!-- PAI-SG-LINKS:END -->

# MM_Fusion_LateFusion

**Late Fusion on AV-MNIST - convergence study (PAI-SG Multimodal Wave 1)**

Part of the *Physical AI* reference series of the **Statera Working Group**. One method of the multimodal convergence comparison in [MM_Wave1A_AVMNIST](https://github.com/Statera-Workshop/MM_Wave1A_AVMNIST).

## Method

Concatenation of the two encoder outputs (48 + 192). *MLP head*: MultiBench MLP 240-100-10. *Linear head*: one linear layer 240-10 (the concat structure of the official OGM-GE code; the unmodulated control for [MM_Fusion_OGMGE](https://github.com/Statera-Workshop/MM_Fusion_OGMGE)).

**Common setup** (identical for the AV-MNIST repositories; full comparison in [MM_Wave1A_AVMNIST](https://github.com/Statera-Workshop/MM_Wave1A_AVMNIST)).
AV-MNIST (MultiBench pre-processed release): 28 x 28 PCA-reduced MNIST digit + 112 x 112 spectrogram of a spoken digit
with ESC-50 background noise, 10 classes; train 55,000 / val 5,000 (last 5,000 of the training file) / test 10,000 pairs.
Encoders from scratch: image LeNet(1,6,3) -> 48-d, audio LeNet(1,6,5) -> 192-d (MultiBench); image standardised with
the training mean / std, audio / 255. Adam 1e-3 constant, batch 64, budgets 30 and 100 epochs, seeds 777 / 778 / 779,
fp32, deterministic cuDNN. Every epoch: test accuracy and a frozen linear probe per modality (10,000 training pairs,
standardised features, full-batch Adam, test accuracy). Reported value = last epoch.

**Convergence measures.** Peak test accuracy and its epoch, drop after the peak (peak - final), and the *audio gap* =
audio-only model's probe - this model's audio probe (same budget; positive = audio under-learned in the fusion model).
*BN re-estimated*: the saved last-epoch weights evaluated again after recomputing BatchNorm statistics on 200 training
batches (the series' evaluation practice); for AV-MNIST it changes nothing beyond +/-0.004.

## Results (AV-MNIST, mean of seeds; range in brackets)

| Model | Optimiser | Epochs | Seeds | Final test acc (range) | Peak (epoch) | Drop after peak | Audio probe | Audio gap | Final, BN re-estimated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Late Fusion (MLP head) | Adam 1e-3 | 30 | 3 | 0.696 (0.000) | 0.722 (8.3) | 0.027 | 0.359 | +0.028 | 0.696 (0.002) |
| Late Fusion (MLP head) | Adam 1e-3 | 100 | 3 | 0.679 (0.004) | 0.722 (8.3) | 0.044 | 0.362 | +0.024 | 0.678 (0.009) |
| Late Fusion (linear head) | Adam 1e-3 | 30 | 3 | 0.712 (0.008) | 0.723 (16.3) | 0.011 | 0.367 | +0.020 | 0.713 (0.008) |
| Late Fusion (linear head) | Adam 1e-3 | 100 | 3 | 0.708 (0.002) | 0.723 (16.3) | 0.015 | 0.380 | +0.005 | 0.708 (0.003) |

Audio-only reference probe: 0.387 (30 epochs) / 0.385 (100 epochs).

**Finding.** Same peak with either head; the MLP head drops much more after the peak than the linear head.

## Weights

Release **v1.0** of this repository (12 files, SHA-256 in `SHA256SUMS.txt`): `avmnist_late_adam_e100_s777.pth`, `avmnist_late_adam_e100_s778.pth`, `avmnist_late_adam_e100_s779.pth`, `avmnist_late_adam_e30_s777.pth`, `avmnist_late_adam_e30_s778.pth`, `avmnist_late_adam_e30_s779.pth`, `avmnist_late_lin_adam_e100_s777.pth`, `avmnist_late_lin_adam_e100_s778.pth`, `avmnist_late_lin_adam_e100_s779.pth`, `avmnist_late_lin_adam_e30_s777.pth`, `avmnist_late_lin_adam_e30_s778.pth`, `avmnist_late_lin_adam_e30_s779.pth`.

```python
import torch, sys; sys.path.insert(0, 'code'); import mm_a as M
m = M.AVNet('<model>')  # img, aud, late, late_lin, tfn, lmf, gblend, ogmge, ogm
m.load_state_dict(torch.load('avmnist_<model>_<opt>_e<epochs>_s<seed>.pth', weights_only=True), strict=True)
```

**Licence note.** The AV-MNIST audio contains ESC-50 noise (CC BY-NC 3.0), so the released weights are for non-commercial use. See THIRD_PARTY.md.

## Run

Data: `gdown 1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp -O avmnist.tar.gz` (MultiBench release), unpack to `~/MM_Wave1/data/avmnist`, check with `python code/count_avmnist.py ~/MM_Wave1/data/avmnist`.

```bash
cd code && bash smoke_a.sh   # gates + 1-epoch smoke
python mm_a.py --model late --epochs 100 --seed 777
python analyze_a.py; python recal_bn_a.py
```
