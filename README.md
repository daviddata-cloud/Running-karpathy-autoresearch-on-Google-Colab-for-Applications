# Running-karpathy-autoresearch-on-Google-Colab-for-Applications
# Technical Log: Adapting `karpathy/autoresearch` for Accessible GPU Environments (Google Colab)

This document details the successful execution of Andrej Karpathy's `autoresearch` framework on a free-tier Google Colab T4 GPU. The goal was to validate this powerful tool for potential "industrialized science research" applications within the public health domain.

The original framework, while groundbreaking, presents several immediate challenges on consumer-grade or cloud-based GPU instances.

### Key Technical Hurdles & Solutions:

1.  **`FlashAttention` Incompatibility:** The default `train.py` script relies on `torch.compile`, which attempts to use FlashAttention. This immediately fails on non-Ampere architecture GPUs (like the Colab T4) with a `RuntimeError`.
    *   **Solution:** The `torch.compile()` step must be disabled. This was achieved by passing the `--no-compile-model` flag.

2.  **CUDA Out of Memory:** The default model parameters (`n_layer=8`, `n_embd=512`, `batch_size=64`) are too large for the ~15GB of VRAM available on a T4 GPU, leading to an immediate `OutOfMemoryError`.
    *   **Solution:** The model's dimensions and batch size must be drastically reduced via command-line arguments to successfully complete a training step.

3.  **Toolchain Argument Passing:** Initial attempts to pass arguments via `uv run train.py --n-layer 2...` failed, proving that the tooling was not forwarding the arguments correctly.
    *   **Solution:** The only reliable method was to bypass `uv run` for execution and call the virtual environment's Python interpreter directly.

### Final Working Command for Google Colab:

This single command, executed after setup, successfully launches the training loop with a compatible configuration.

```bash
# Execute directly from the venv's python, disable compilation, and reduce model size
!.venv/bin/python train.py --compile-model=False --n-layer=2 --n_embd=256 --batch-size=8
```

This hands-on validation proves the feasibility of using this PhD-level research automation framework on accessible hardware, opening the door for rapid, domain-specific knowledge engineering in areas like public health.

<img width="1079" height="620" alt="image" src="https://github.com/user-attachments/assets/fe201d6e-468a-411e-8180-cb5db395b520" />
