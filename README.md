# mat-pref

Mat-Pref is a benchmark of **10,837 ionic-substitution questions** across **11 inorganic structure families**, with every candidate answer backed by a first-principles Materials Project DFT calculation. Three evaluation splits isolate distinct generalization axes:

| Split | Logical Qs | Question instances | Tests |
|-------|------------|--------------------|-------|
| IID | 89 | 356 | New hosts within training families |
| OOD-host | 282 | 700 | Entirely held-out families (garnet, halide perovskite, NASICON) |
| OOD-property | 510 | 2,278 | Held-out band-gap property on training-family hosts |

See the paper for full benchmark construction, training protocol (SFT → GRPO on Qwen3-8B), and per-family results.

## Dataset

The benchmark is released on Hugging Face Datasets:

```python
from datasets import load_dataset

ds = load_dataset("agarosegirls/mat-pref")
ds["test_iid"], ds["test_ood_host"], ds["test_ood_property"]
```

Trained checkpoints (LoRA adapters over Qwen3-8B):

```python
from huggingface_hub import snapshot_download

sft_path  = snapshot_download("agarosegirls/mat-pref-qwen3-8b-sft")
grpo_path = snapshot_download("agarosegirls/mat-pref-qwen3-8b-grpo")
```

## Install

```bash
git clone https://github.com/<your-org>/mat-pref.git
cd mat-pref
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Evaluating your own model

`scripts/run_baselines.py` runs zero-shot, few-shot, and chemical-context regimes against any OpenAI-compatible endpoint (the paper used DeepInfra for Qwen2.5-72B, Llama 3.3-70B, DeepSeek-V3, and Qwen3-235B-Instruct).

```bash
export DEEPINFRA_API_KEY=...

python scripts/run_baselines.py --model qwen3-235b --split test_ood_host
```

Wrap your own evaluator around the released JSONL splits to score against the DFT-backed ground truth.

## Reproducing the training

The two-stage SFT → GRPO pipeline uses [Tinker](https://thinkingmachines.ai/blog/announcing-tinker/) on a single Qwen3-8B (or Llama-3.1-8B) base. Total pipeline cost in the paper was under $50.

```bash
# 1. (Optional) regenerate SFT traces with your own teacher model
python scripts/generate_traces.py

# 2. SFT
python training/tinker_sft.py        # Qwen3-8B
python training/tinker_sft_llama.py  # Llama-3.1-8B replication

# 3. GRPO from the SFT checkpoint
python training/tinker_grpo.py
python training/tinker_grpo_llama.py

# 4. Evaluate
python training/tinker_eval.py
python training/tinker_eval_llama.py
```

Hyperparameters are set inline in each training script (matching the values reported in the paper).

## Citation

```bibtex
@inproceedings{leung2026matpref,
  title     = {Mat-Pref: Verifiable-Reward Training Improves Compositional Reasoning in Inorganic Materials},
  author    = {Leung, Sarrah Mikhail and Kim, Taehan and Park, Jeongbin},
  booktitle = {ICML AI4Physics Workshop},
  year      = {2026},
}
```
