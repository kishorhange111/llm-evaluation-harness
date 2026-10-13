"""Correctness tests for the Hugging Face backend using a tiny randomly initialised GPT-2.

Everything is built locally (no downloads). We check the two things most
likely to be silently wrong: that batched/padded log-likelihoods equal a
straightforward one-at-a-time computation, and that left-padded batched
generation equals unbatched generation.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from evalkit.models.hf import HFModel  # noqa: E402
from evalkit.types import GenerationRequest, LoglikelihoodRequest  # noqa: E402

WORDS = "the cat sat on mat a dog ran to park answer is yes no question".split()


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Save a 2-layer GPT-2 with a word-level tokenizer to a temp dir, like a Hub checkpoint."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, **{w: i + 2 for i, w in enumerate(WORDS)}, "[UNK]": len(WORDS) + 2}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="[UNK]")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=len(vocab), n_positions=64, n_embd=32, n_layer=2, n_head=2, pad_token_id=0, eos_token_id=1)
    )
    path = tmp_path_factory.mktemp("tiny")
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return str(path)


def manual_logprob(m: HFModel, context: str, continuation: str) -> float:
    """Reference implementation: one unpadded forward pass, sum continuation log-probs."""
    ctx, cont = m._encode_pair(context, continuation)
    ids = torch.tensor([ctx + cont])
    with torch.no_grad():
        lp = torch.log_softmax(m.model(ids).logits.float(), -1)[0]
    return sum(float(lp[len(ctx) + i - 1, t]) for i, t in enumerate(cont))


def test_batched_loglikelihood_matches_reference(tiny_model_dir):
    m = HFModel(tiny_model_dir, batch_size=3, dtype="float32", device_map="cpu")
    reqs = [
        LoglikelihoodRequest("the cat sat on", " mat"),
        LoglikelihoodRequest("a dog ran to the", " park yes"),
        LoglikelihoodRequest("question answer is", " no"),
        LoglikelihoodRequest("the", " cat sat on a mat"),
    ]
    got = m.loglikelihood(reqs)
    for r, res in zip(reqs, got):
        assert res.logprob == pytest.approx(manual_logprob(m, r.context, r.continuation), abs=1e-4)
    assert [g.num_tokens for g in got] == [1, 2, 1, 5]


def test_batched_generation_matches_unbatched(tiny_model_dir):
    m = HFModel(tiny_model_dir, batch_size=4, dtype="float32", device_map="cpu")
    prompts = ["the cat", "a dog ran to the park", "question"]
    batched = m.generate([GenerationRequest(p, max_new_tokens=5) for p in prompts])
    m.batch_size = 1
    single = m.generate([GenerationRequest(p, max_new_tokens=5) for p in prompts])
    assert batched == single
