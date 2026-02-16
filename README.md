---
title: Voice Detection Api
emoji: 🐠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: AI vs Human Voice Detection
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
Our commit repo for final submission: https://huggingface.co/spaces/arshan123/voice-detection-api/tree/main

<img width="1489" height="863" alt="{5E6DC937-C9D2-4548-877A-599F0F592098}" src="https://github.com/user-attachments/assets/d0979783-619d-4aa9-85d8-48e92be8d44a" />


## Reducing latency on Hugging Face Spaces

- **Hardware:** In the Space **Settings → Hardware**, choose **Nvidia T4 small** (~$0.40/hr). This typically cuts inference from ~30s (CPU) to ~2–4s.
- **Code:** The app already uses the GPU when available and loads models in **float16** on CUDA for faster T4 inference. No extra code changes are required.
- **Cost:** Billing is per minute. Pause the Space when you’re not demoing or coding to avoid charges.
