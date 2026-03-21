blocks/
├── __init__.py
├── attention.py      ← MHA block (one transformer layer)
├── transformer.py    ← stack of MHA layers = full backbone
├── mamba2.py         ← SSM block + stack = full backbone
├── consistency.py    ← head MLP
└── embed.py          ← two things:
                         1. PatchEmbed: video latents → tokens
                         2. TextEmbed: captions → embeddings (T5)
