╔══════════════════════════════════════════════════════════════╗
║  [1] 👁️  Vision only   — SigLIP-2 encoder + MLP projector  ║
║  [2] 🎤 Audio only    — Conformer encoder + MLP projector  ║
║  [3] 🌐 Omni (both)   — Vision + Audio in one GGUF         ║
╚══════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════╗
║  Phi-4 Multimodal mmproj — Mode Selection                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [1] 👁️  Vision only   — SigLIP-2 encoder + MLP projector  ║
║      Image understanding, VQA, chart reading                ║
║                                                              ║
║  [2] 🎤 Audio only    — Conformer encoder + MLP projector  ║
║      Speech recognition (ASR), translation, summarization  ║
║                                                              ║
║  [3] 🌐 Omni (both)   — Vision + Audio in one GGUF         ║
║      Full multimodal: images AND speech                     ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  ℹ️  Compatibility                                           ║
║                                                              ║
║  👁️  Vision works with mainline llama.cpp (llama-server).    ║
║                                                              ║
║  🎤 Audio requires C++ Conformer support.                   ║
║     Use this fork with audio support:                       ║
║     https://github.com/Ahmed-Shayan-Arsalan/               ║
║       Phi4-multimodal-Quantisized-Llama.cpp                 ║
║     Or wait for mainline llama.cpp to merge Conformer PRs.  ║
║                                                              ║
║  🌐 Omni GGUF: vision works immediately, audio once         ║
║     Conformer support lands in your llama.cpp build.        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
