# RealFormer

PyTorch implementation of [RealFormer: Transformer Likes Residual Attention](https://arxiv.org/abs/2012.11747).

## Quickstart

Clone this repository.

```
git clone https://github.com/jaketae/realformer.git
```

Navigate to the cloned directory. You can start using the model via

```python
>>> from realformer import RealFormerEncoder
>>> model = RealFormerEncoder()
```

By default, the model comes with the following parameters:

```python
RealFormerEncoder(
    d_model=512,
    num_heads=8,
    expansion_factor=2,
    dropout=0.5,
    max_len=512,
    num_layers=6,
)
```

## Summary

Residual Attention Layer Transformer, shortened as RealFormer, is a transformer variant that incorporatess residual skip connections to allow previous attention scores to pass through the entire network. It outperforms canonical transformers on a variety of tasks and datasets, including masked language modeling (MLM), [GLUE](https://gluebenchmark.com), and [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).

## Implementation Notes

-   Just like `torch.nn.TransformerEncoder`, the `RealFormerEncoder` does not include any embedding layers. It is recommended that you implemenet positional encoding schemes (e.g. sinusodial tables, learnable embeddings) as needed.
-   The authors mention that RealFormer layers can be used as drop-in replacements for any transformer model, whether they be autoencoding (encoders) or auto-regressive (decoders). We closely follow the flow of the paper and include only an encoder version of the implementation for now.

## Resources

-   [Original Paper](https://arxiv.org/abs/2012.11747)
