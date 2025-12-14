# Contributing to OpenLabLM

Thank you for considering contributing to OpenLabLM! This project exists to prove that the hobbyist community can build, understand, and improve modern LLMs. We welcome everyone, from seasoned ML engineers to first-time coders. Even if it's your first pull request, it can never hurt to offer help :D

## How to Contribute

There are two primary ways to contribute to OpenLabLM:

### 1. The "Community Choice" (Reddit)
As stated in our README, we run an experiment where high-level decisions or specific feature additions can be driven by community consensus.
- Watch for periodic threads on r/LocalLLaMA.
- Suggest changes or features in the comments.
- **The top upvoted technical suggestion** is reviewed and implemented by the maintainers at the end of the cycle.

### 2. GitHub Pull Requests
For direct code contributions, bug fixes, and immediate improvements, please submit a Pull Request (PR).

If you are submitting a Pull Request please describe the change and include some validation loss data if applicable

#### Workflow:
1.  **Fork** the repository.
2.  **Clone** the project to your local machine.
3.  **Create a new branch** (`git checkout -b feature/AmazingFeature`).
4.  **Commit** your changes.
5.  **Push** to the branch.
6.  **Open a Pull Request**.

## Design Philosophy & Guidelines

To keep this project accessible and functional for hobbyists, we strictly enforce a few constraints. Please ensure your contributions adhere to these:

*   **8GB VRAM:** The core goal is accessibility. If your feature pushes memory usage too far beyond 8GBs of VRAM on a standard consumer GPU, it should be optional or optimized. Realistically staying within 24GBs of VRAM is acceptable too, but at a certain point it is simply out of scope.
*   **Readability over Micro-Optimization:** This is a learning tool. Code should be "hackable" and easy to understand. Comments are encouraged!!
*   **Justificaiton:** If you are changing core components, please provide an ablation or justification (e.g., "This uses less VRAM" or "This converges faster").

## Reporting Bugs

If you find a bug, please create an Issue using the provided template. Include:
*   Your GPU/Hardware specs.
*   The dataset you were using (e.g., Tiny-Shakespeare, Dolly15k).
*   Steps to reproduce the error.

## Dataset Contributions

If you are adding support for a new dataset:
0. Ensure there is no harmful or copyrighted content.
1.  Ensure the loader is efficient.
2.  Verify it works with the existing tokenizer or include the necessary tokenizer updates.
3.  Add a reference to the dataset source in the README.


Thank you for building with us!