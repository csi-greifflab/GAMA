# Attribution assignment for deep-generative sequence models enables interpretability analysis using positive-only data

We introduce **Generative Attribution Metric Analysis (GAMA)**, the first gradient-based attribution framework tailored to autoregressive generative networks trained only on positive (one-class) sequences.

This repository demonstrates how to compute GAMA (Generative Attribution Metric Analysis) as presented in our paper. Our work achieves the following:

* Bridges an unmet methodological gap by adapting Integrated Gradients to the generative setting, quantifying token-wise importance across an entire learned distribution.
* Demonstrates rigorous performance across 270 synthetic benchmarks that vary noise, motif placement, and higher-order logic, recovering signal motifs with <5% false-negative rate at realistic noise levels.
* Correlates strongly with biophysical ground truth in antibody-antigen simulations (Spearman ρ = 0.74, p = 0.004) and correctly pinpoints three of four key residues in an experimental antibody binding dataset, validating real-world utility.
* Paves the way for trustworthy generative design by revealing why a model proposes specific sequences-critical for sensitive applications such as therapeutic antibody lead discovery.
  

📄 Preprint available: [arxiv](https://arxiv.org/abs/2506.23182)
