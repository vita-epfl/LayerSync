
## 🚀 Quick Start

To test LayerSync for fine-tuning WAN with LoRA, follow these steps to integrate our modifications into the DiffSynth-Studio framework.

### 1. Environment Setup

First, clone the [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio.git) repository and follow their instructions to set up the environment, download required checkpoints, and prepare your dataset.

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
# Follow DiffSynth-Studio setup instructions here

```

### 2. Integrate LayerSync

Replace the following core files in the `DiffSynth-Studio` directory with the optimized versions provided in this repository:

| Original File Path | Replacement Action |
| --- | --- |
| `diffsynth/diffusion/loss.py` | Replace with our `loss.py` |
| `diffsynth/pipelines/wan_video.py` | Replace with our `wan_video.py` |
| `examples/wanvideo/model_training/train.py` | Replace with our `train.py` |

### 3. Start Training

To enable LayerSync during fine-tuning, use the `--task "sft_layersync"` flag when running the training script:

```bash
python examples/wanvideo/model_training/train.py \
    --task "sft_layersync" \
    --dataset_base_path /path/to/your/data \
    --dataset_metadata_path /path/to/your/metadata \
    --output_path ./output_models

```

---

## 📺 Comparisons (WAN2.1 1B T2I)

The following results demonstrate the performance of the WAN2.1 1B Text-to-Image model fine-tuned on the **Mixkit dataset**. LayerSync shows significantly better alignment and visual consistency compared to vanilla LoRA fine-tuning.

| Vanilla Fine-tuning | LayerSync Fine-tuning (Ours) |
| --- | --- |
| [Placeholder: Vanilla Video 1] | [Placeholder: LayerSync Video 1] |
| [Placeholder: Vanilla Video 2] | [Placeholder: LayerSync Video 2] |
| [Placeholder: Vanilla Video 3] | [Placeholder: LayerSync Video 3] |
| [Placeholder: Vanilla Video 4] | [Placeholder: LayerSync Video 4] |
| [Placeholder: Vanilla Video 5] | [Placeholder: LayerSync Video 5] |

