"""Exact regression test for direct paged GDN convolution checkpoints."""

import torch

from gllm.layers.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_update,
    causal_conv1d_update_paged,
)


def main():
    torch.manual_seed(7)
    for batch in (1, 7):
        for width in (2, 3, 4):
            qlen, dim = 4, 513
            num_blocks = batch * qlen + 3
            x = torch.randn(
                batch, dim, qlen, device="cuda", dtype=torch.bfloat16
            )
            weight = torch.randn(
                dim, width, device="cuda", dtype=torch.bfloat16
            )
            bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
            initial = torch.randn(
                num_blocks,
                dim,
                width - 1,
                device="cuda",
                dtype=torch.bfloat16,
            )
            block_table = torch.arange(
                1,
                1 + batch * qlen,
                device="cuda",
                dtype=torch.int32,
            ).view(batch, qlen)
            accepted = torch.randint(
                1, qlen + 1, (batch,), device="cuda", dtype=torch.int32
            )

            # Recreate the former gather -> wide update -> scatter path using
            # torch indexing.  The update kernel remains the numerical oracle,
            # so the new fusion must be bit-identical rather than merely close.
            rows = torch.arange(batch, device="cuda")
            source = block_table[rows, accepted.long() - 1].long()
            scratch = torch.zeros(
                batch,
                dim,
                width - 1 + qlen - 1,
                device="cuda",
                dtype=torch.bfloat16,
            )
            scratch[:, :, : width - 1].copy_(initial.index_select(0, source))
            intermediate = torch.empty(
                batch,
                qlen,
                dim,
                width - 1,
                device="cuda",
                dtype=torch.bfloat16,
            )
            identity = torch.arange(batch, device="cuda", dtype=torch.int32)
            reference = causal_conv1d_update(
                x,
                scratch,
                weight,
                bias,
                "silu",
                conv_state_indices=identity,
                num_accepted_tokens=torch.ones(
                    batch, device="cuda", dtype=torch.int32
                ),
                intermediate_conv_window=intermediate,
                intermediate_state_indices=identity,
            )
            reference_state = initial.clone()
            reference_state[block_table.long().reshape(-1)] = intermediate.reshape(
                batch * qlen, dim, width - 1
            )

            actual_state = initial.clone()
            actual = causal_conv1d_update_paged(
                x,
                actual_state,
                weight,
                block_table,
                accepted,
                bias=bias,
                activation="silu",
            )
            checkpoint_ids = block_table.long().reshape(-1)
            assert torch.equal(actual, reference), (batch, width)
            assert torch.equal(
                actual_state.index_select(0, checkpoint_ids),
                reference_state.index_select(0, checkpoint_ids),
            ), (batch, width)

    print("paged GDN conv exactly matches staged reference")


if __name__ == "__main__":
    main()
