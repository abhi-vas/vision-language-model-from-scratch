## PaliGemma Implementation

Implemented the core components of **PaliGemma from scratch in PyTorch**, covering the vision encoder, language model, multimodal integration, preprocessing, and inference pipeline.

### Core Components

- **`modeling_siglip.py`** — Implements the SigLIP vision encoder to convert images into visual embeddings.

- **`modeling_gemma.py`** — Implements the Gemma Transformer language model for processing text and visual embeddings.

- **Multimodal Projector** — Projects SigLIP visual embeddings into Gemma's embedding space to enable vision-language integration.

- **`PaliGemmaForConditionalGeneration`** — Integrates SigLIP, the multimodal projector, and Gemma into a unified vision-language model.

- **`processing_paligemma.py`** — Handles image preprocessing, text tokenization, image tokens, attention masks, and model input preparation.

- **`inference.py`** — Implements autoregressive generation with next-token prediction and KV caching for efficient inference.
