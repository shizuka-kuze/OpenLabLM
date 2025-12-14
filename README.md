# Can Hobbyists Build an LLM?

The point of this project is to see if Reddit users on r/LocalLLaMA and other hobbyist communities can design an effective LLM. The theory is that there is great potential within the open-source and hobbyist community to run ablations, generate ideas, provide insights, and incorporate learnings extremely quickly.

This repo is an implementation of an LLM, like ChatGPT, in a single, hackable, dependency-lite but initially very standard codebase. It's designed to run on a single consumer GPU. The goal is to run within 8GB of VRAM to allow the greatest range of exploration and participation.

## Demo

*Training and Validation Conducted with Tiny Shakespeare on a single consumer laptop GPU*
1) Each epoch takes around 10 seconds!
2) After a few epchs results look decent!
3) First epoch is the slowest due to cold start.

https://github.com/user-attachments/assets/0485e4fe-2cda-49ad-9e4b-20c81208abc7

#### Run the Code
```bash
git clone https://github.com/shizuka-kuze/OpenLabLM.git
cd OpenLabLM
conda activate openlablm
pip install -r requirements.txt
python main.py
```

## Features

The LLM uses minimal implementations of the following:
- ReLU² Blocks
- RMS Normalization
- The Muon Optimizer
- Multi-Head Latent Attention
- Rotary Positional Embeddings
- Data Loading, Validation, and Evaluation 
- Support for many datasets!

## Contributing

**The top upvoted comment will be added at the end of every day!!**

Contributions through pull requests are welcome too.

See `contributing.md` for ways to get started.

Please adhere to this project's `Code of Conduct`.
#### Ethics

This project is contingent on the ideas of advancing the pursuit of human knowledge through free, open-source, and accountable software. Transparency is provided through available, non-obfuscated code that anyone can run.

#### Motivation
We hope to expand community knowledge and cooperation, and demonstrate the advantage of open projects.

#### Copyright and Data Usage
The datasets used are a mix of real and synthetic data designed specifically for LLM training. Particularly: 

 - [Tiny Shakespeare (small training)](https://huggingface.co/datasets/karpathy/tiny_shakespeare)
 - [Databricks Dolly 15k (mid-level training)](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
 - [HuggingFaceTB SmolLM Corpus (larger scale training)](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)

## Changelog

- 12/13/25 Initial Barebones Commit

## Contact 


For contact or coordination email deepseoul@proton.me or join our [Slack channel.]()
