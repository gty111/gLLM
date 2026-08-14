"""Attention layer building blocks and kernel backends.

The modules are intentionally separated by responsibility:

* :mod:`base` provides model-side head geometry shared by QKV projection layers.
* :mod:`qkv` owns explicit-QKV MHA/GQA/MQA cache writes and dispatch.
* :mod:`qkv_backends` implements the FA4 and FlashInfer kernel adapters.
* :mod:`mla` owns the independent MLA execution path.
"""
