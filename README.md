## PaliGemma Implementation

Implemented the core components of **PaliGemma from scratch in PyTorch**, covering the vision encoder, language model, multimodal integration, preprocessing, and inference pipeline.

## Essential Files

The following files contain the core components required to understand and implement PaliGemma:

- **modeling_siglip.py** — Implements the SigLIP vision encoder for converting images into visual embeddings.

- **modeling_gemma.py** — Implements the Gemma Transformer language model for processing text and visual embeddings.

- **processing_paligemma.py** — Handles image preprocessing, text tokenization, image tokens, attention masks, and model input preparation.

- **inference.py** — Implements autoregressive generation and KV caching for efficient inference.

- **utils.py** — Provides supporting utility functions required by the implementation.


