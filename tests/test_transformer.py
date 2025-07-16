import pytest
import torch
import torch.nn as nn

# Assumes MHAttention is in a file named mha.py in the same directory
from broccoli.transformer import MHAttention

# Constants for testing
EMBED_DIM = 12
N_HEADS = 4
SEQ_LEN = 10
BATCH_SIZE = 2
ATOL = 1e-6  # Absolute tolerance for tensor comparisons


@pytest.fixture
def dummy_tensors():
    """Provides a standard set of tensors for attention tests."""
    query = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    key = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    value = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    return query, key, value


def test_non_causal_attention_matches_pytorch(dummy_tensors):
    """
    Verifies that the custom non-causal attention matches the output of
    torch.nn.MultiheadAttention when weights are identical.
    """
    query, key, value = dummy_tensors

    # 1. Instantiate PyTorch's implementation
    pytorch_mha = nn.MultiheadAttention(
        embed_dim=EMBED_DIM, num_heads=N_HEADS, bias=False, batch_first=True
    )

    # 2. Instantiate our custom implementation
    custom_mha = MHAttention(
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        share_kv=False,  # Must be False to match PyTorch's separate k,v projections
    )

    # 3. Copy weights from PyTorch MHA to custom MHA for a fair comparison
    q_w, k_w, v_w = pytorch_mha.in_proj_weight.chunk(3)
    with torch.no_grad():
        custom_mha.q_proj.weight.copy_(q_w)
        custom_mha.k_proj.weight.copy_(k_w)
        custom_mha.v_proj.weight.copy_(v_w)
        custom_mha.out_proj.weight.copy_(pytorch_mha.out_proj.weight)

    # 4. Get outputs from both models
    output_pytorch, _ = pytorch_mha(query, key, value)
    output_custom = custom_mha(query, key, value)

    # 5. Assert that the outputs are numerically close
    assert output_pytorch.shape == output_custom.shape
    assert torch.allclose(output_pytorch, output_custom, atol=ATOL)


def test_causal_attention_matches_pytorch(dummy_tensors):
    """
    Verifies that the custom causal attention matches the output of
    torch.nn.MultiheadAttention when using a causal mask.
    """
    query, key, value = dummy_tensors

    # 1. Instantiate PyTorch's implementation
    pytorch_mha = nn.MultiheadAttention(
        embed_dim=EMBED_DIM, num_heads=N_HEADS, bias=False, batch_first=True
    )

    # 2. Instantiate our custom implementation for causal attention
    custom_mha = MHAttention(
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        causal=True,
        sequence_length=SEQ_LEN,
        share_kv=False,
    )

    # 3. Copy weights for a direct comparison
    q_w, k_w, v_w = pytorch_mha.in_proj_weight.chunk(3)
    with torch.no_grad():
        custom_mha.q_proj.weight.copy_(q_w)
        custom_mha.k_proj.weight.copy_(k_w)
        custom_mha.v_proj.weight.copy_(v_w)
        custom_mha.out_proj.weight.copy_(pytorch_mha.out_proj.weight)

    # 4. Get outputs from both, ensuring PyTorch version uses the causal flag
    # Note: is_causal=True is the modern way to enable causal masking in PyTorch
    output_pytorch, _ = pytorch_mha(
        query,
        key,
        value,
        is_causal=True,
        attn_mask=torch.nn.Transformer.generate_square_subsequent_mask(SEQ_LEN),
    )
    output_custom = custom_mha(query, key, value)

    # 5. Assert that the outputs are numerically close
    assert output_pytorch.shape == output_custom.shape
    assert torch.allclose(output_pytorch, output_custom, atol=ATOL)


def test_shared_kv_projection():
    """Checks that k_proj and v_proj are the same module when share_kv is True."""
    mha = MHAttention(EMBED_DIM, N_HEADS, share_kv=True)
    # Check if they are the exact same object in memory
    assert mha.k_proj is mha.v_proj


def test_dropout_is_active(dummy_tensors):
    """Ensures dropout is applied during training and disabled during evaluation."""
    query, key, value = dummy_tensors

    mha_with_dropout = MHAttention(
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        dropout=0.5,  # Use a high dropout rate to ensure outputs differ
    )

    # In training mode, dropout should be active
    mha_with_dropout.train()
    output1_train = mha_with_dropout(query, key, value)
    output2_train = mha_with_dropout(query, key, value)
    # Outputs should NOT be the same due to random dropout
    assert not torch.equal(output1_train, output2_train)

    # In evaluation mode, dropout should be disabled
    mha_with_dropout.eval()
    output1_eval = mha_with_dropout(query, key, value)
    output2_eval = mha_with_dropout(query, key, value)
    # Outputs SHOULD be the same
    assert torch.equal(output1_eval, output2_eval)


def test_causal_assertion_error():
    """Ensures that creating a causal model without a sequence_length raises an error."""
    with pytest.raises(AssertionError):
        MHAttention(
            embed_dim=EMBED_DIM,
            n_heads=N_HEADS,
            causal=True,
            sequence_length=None,  # This should trigger the assertion
        )
