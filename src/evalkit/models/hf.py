"""Hugging Face ``transformers`` backend: runs a model locally on GPU/CPU.

Two subtleties that are easy to get wrong, and that silently change scores:

1. **Padding side.** For *generation* we pad on the left, so every prompt ends
   right where generation starts. For *log-likelihood* we pad on the right and
   read logits at the real token positions.
2. **Tokenisation boundary.** ``tok(context + continuation)`` is not always
   ``tok(context) + tok(continuation)`` (BPE merges across the boundary). We
   tokenise the joined string and take the continuation tokens as everything
   after the context's tokens, which matches what the model actually sees.
"""

from __future__ import annotations

from typing import Any

from evalkit.models.base import Model, truncate_at_stop
from evalkit.types import GenerationRequest, LoglikelihoodRequest, LoglikelihoodResult


class HFModel(Model):
    """Local model loaded with ``AutoModelForCausalLM``.

    Args:
        pretrained: Hub id or local path, e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``.
        batch_size: Requests per forward pass. Bigger is faster until GPU memory runs out.
        dtype: ``"bfloat16"`` on Ampere+ GPUs (A100/L4), ``"float16"`` on T4, ``"float32"`` on CPU.
        chat: If True, prompts are wrapped with the tokenizer's chat template.
        system_prompt: Optional system message used with ``chat=True``.
        device_map: Passed to ``from_pretrained``; ``"auto"`` spreads across available GPUs.
    """

    def __init__(
        self,
        pretrained: str,
        batch_size: int = 16,
        dtype: str = "bfloat16",
        chat: bool = False,
        system_prompt: str | None = None,
        device_map: str = "auto",
        max_length: int | None = None,
        **model_kwargs: Any,
    ):
        # Imported lazily so the rest of the package (and its tests) work without torch installed.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = pretrained
        self.batch_size = batch_size
        self.chat = chat
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        # Many causal LMs ship without a pad token; reusing EOS is the standard fix
        # and is harmless because padded positions are masked out.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained, torch_dtype=getattr(torch, dtype), device_map=device_map, **model_kwargs
        )
        self.model.eval()  # disables dropout: evaluation must be deterministic
        self.max_length = max_length or getattr(self.model.config, "max_position_embeddings", 4096)

    # ------------------------------------------------------------------ prompts
    def apply_chat_template(self, user_message: str) -> str:
        """Render a single user turn with the model's own chat template (if ``chat=True``)."""
        if not self.chat:
            return user_message
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})
        # add_generation_prompt=True appends the "<assistant>" header so the model answers
        # instead of continuing the user's message.
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # --------------------------------------------------------------- generation
    def generate(self, requests: list[GenerationRequest]) -> list[list[str]]:
        torch = self.torch
        tok = self.tokenizer
        tok.padding_side = "left"  # see module docstring, point 1
        outputs: list[list[str]] = [[] for _ in requests]

        # The runner sends one task's requests at a time, and a task uses the same
        # decoding settings (n, temperature) for all of them, so a batch can share them.
        for start in range(0, len(requests), self.batch_size):
            batch = requests[start : start + self.batch_size]
            cfg = batch[0]
            # A rendered chat template already contains BOS/special tokens; don't add them twice.
            enc = tok([r.prompt for r in batch], return_tensors="pt", padding=True, add_special_tokens=not self.chat).to(self.model.device)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max(r.max_new_tokens for r in batch),
                "pad_token_id": tok.pad_token_id,
                "num_return_sequences": cfg.n,
            }
            if cfg.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=0.95)
            else:
                gen_kwargs.update(do_sample=False)
            stops = sorted({s for r in batch for s in r.stop})
            if stops:
                # Let generate() stop as soon as any stop string appears, saving compute.
                gen_kwargs.update(stop_strings=stops, tokenizer=tok)

            with torch.inference_mode():
                out = self.model.generate(**enc, **gen_kwargs)

            # Drop the prompt tokens: with left padding every prompt has the same length.
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for i, req in enumerate(batch):
                samples = texts[i * cfg.n : (i + 1) * cfg.n]
                outputs[start + i] = [truncate_at_stop(t, req.stop) for t in samples]
        return outputs

    # ----------------------------------------------------------- loglikelihood
    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Tokenise a (context, continuation) pair; see module docstring, point 2."""
        # Move trailing spaces from the context to the continuation ("Answer: " + "B" ->
        # "Answer:" + " B"), because most tokenizers attach the space to the next word.
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole = self.tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
        ctx = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        return ctx, whole[len(ctx) :]

    def loglikelihood(self, requests: list[LoglikelihoodRequest]) -> list[LoglikelihoodResult]:
        torch = self.torch
        pairs = [self._encode_pair(r.context, r.continuation) for r in requests]
        # Sort longest-first so each batch has similar lengths (less padding waste),
        # and an out-of-memory error shows up in the very first batch rather than hours in.
        order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        results: list[LoglikelihoodResult | None] = [None] * len(pairs)

        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            seqs, cont_lens = [], []
            for i in idx:
                ctx, cont = pairs[i]
                # Keep the last max_length tokens (the continuation must never be cut);
                # the final token is not fed in because nothing predicts after it.
                seq = (ctx + cont)[-(self.max_length + 1) :]
                seqs.append(seq[:-1])
                cont_lens.append(len(cont))
            width = max(len(s) for s in seqs)
            pad = self.tokenizer.pad_token_id
            input_ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=self.model.device)
            attn = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=self.model.device)

            with torch.inference_mode():
                logprobs = torch.log_softmax(self.model(input_ids=input_ids, attention_mask=attn).logits.float(), dim=-1)

            for row, i in enumerate(idx):
                ctx, cont = pairs[i]
                n = cont_lens[row]
                end = len(seqs[row])  # logits at position t predict token t+1
                # Logits at positions [end-n, end) predict the n continuation tokens.
                lp = logprobs[row, end - n : end]
                target = torch.tensor(cont, device=lp.device)
                token_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                greedy = bool((lp.argmax(-1) == target).all())
                results[i] = LoglikelihoodResult(float(token_lp.sum()), n, greedy)
        return results  # type: ignore[return-value]
