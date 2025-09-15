# Reformer++: An Efficient Transformer for Long-Sequence Modeling

This repository presents an experimental Transformer framework designed to tackle the **memory and computational bottlenecks of traditional transformers**. At its core is a novel architecture: Reformer++. 

This new model is engineered with **learnable Locality-Sensitive Hashing (LSH), hybrid local-global routing, and gated reversible layers**, making it ideal for efficient, long-sequence tasks like language modeling and translation.



## What is Reformer and Reformer++?

**Reformer** is an efficient Transformer model designed to handle long sequences with less memory and computation. It achieves this using two core techniques:
- **Locality-Sensitive Hashing (LSH):** Approximates the costly dot-product attention, reducing its complexity from quadratic to O(LlogL).
- **Reversible Layers:** Enables memory-efficient training by allowing activations to be re-computed during backpropagation instead of stored, saving significant memory.

**Reformer++** builds on this foundation by introducing several key innovations to further improve performance and flexibility for complex language tasks:
- **Learnable LSH:** Makes the bucket assignment trainable, allowing the model to adapt and find more optimal attention patterns.
- **Hybrid Local-Global Routing:** Combines sparse attention with a new routing mechanism to maintain crucial long-range dependencies.
- **Gated Reversible Blocks:** A new type of reversible layer that enhances the model's capacity and learning ability.

All code and research findings are currently being prepared and will be available shortly.
