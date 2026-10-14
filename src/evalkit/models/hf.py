"""Hugging Face ``transformers`` backend: runs a model locally on GPU/CPU.

Two subtleties that are easy to get wrong, and that silently change scores:

1. **Padding side.** We pad on the left everywhere, so every sequence ends at
   the same column. For generation that is where new tokens start; for
   log-likelihood it means the continuation tokens of every row sit in the
   last few columns, so the model only has to project those few positions onto
   the vocabulary (``logits_to_keep``). Projecting every position would need
   batch x length x 152k-vocab floats, tens of GB for 5-shot MMLU.
2. **Tokenisation boundary.** ``tok(context + continuation)`` is not always
   ``tok(context) + tok(continuation)`` (BPE merges across the boundary). We
   tokenise the joined string and take the continuation tokens as everything
   after the context's tokens, which matches what the model actually sees.
3. **BOS token.** Llama/Gemma-style models expect a beginning-of-sequence token
   and score worse without it; Qwen/GPT-2 have none. We prepend whatever the
   tokenizer itself adds to an empty string, and use EOS as the context when
   the context is empty (so the first continuation token has something to be predicted from).
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
        max_batch_tokens: int = 32_768,
        **model_kwargs: Any,
    ):
        # Imported lazily so the rest of the package (and its tests) work without torch installed.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = pretrained
        self.batch_size = batch_size
        # Log-likelihood batches are also capped by total tokens, so a batch of long 5-shot prompts doesn't run out of memory.
        self.max_batch_tokens = max_batch_tokens
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
        # Special tokens the tokenizer adds on its own (e.g. Llama's <s>); empty for Qwen and GPT-2.
        self._prefix: list[int] = list(self.tokenizer("", add_special_tokens=True)["input_ids"])

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
        cont = whole[len(ctx) :]
        ctx = self._prefix + ctx
        if not ctx:  # nothing would predict the first continuation token
            ctx = [self.tokenizer.eos_token_id]
        return ctx, cont

    def _batches(self, order: list[int], lengths: list[int]):
        """Group indices (already sorted longest first) by batch size and by a total-token budget."""
        batch: list[int] = []
        for i in order:
            # Every row is padded to the first (longest) row's length.
            width = lengths[batch[0]] if batch else lengths[i]
            if batch and (len(batch) >= self.batch_size or (len(batch) + 1) * width > self.max_batch_tokens):
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    def _continuation_logits(self, input_ids, attn, keep: int):
        """Logits for only the last ``keep`` positions (left padding puts every continuation there)."""
        # Explicit positions so left padding doesn't shift absolute position embeddings (GPT-2 style).
        position_ids = (attn.cumsum(-1) - 1).clamp(min=0)
        kwargs = {"input_ids": input_ids, "attention_mask": attn, "position_ids": position_ids}
        try:
            return self.model(**kwargs, logits_to_keep=keep).logits[:, -keep:]
        except TypeError:  # older architectures without logits_to_keep: project everything, then slice
            return self.model(**kwargs).logits[:, -keep:]

    def loglikelihood(self, requests: list[LoglikelihoodRequest]) -> list[LoglikelihoodResult]:
        torch = self.torch
        pairs = [self._encode_pair(r.context, r.continuation) for r in requests]
        # Keep the last max_length tokens (the continuation must never be cut);
        # the final token is not fed in because nothing predicts after it.
        seqs_all = [(ctx + cont)[-(self.max_length + 1) :][:-1] for ctx, cont in pairs]
        lengths = [len(s) for s in seqs_all]
        # Sort longest-first so each batch has similar lengths (less padding waste),
        # and an out-of-memory error shows up in the very first batch rather than hours in.
        order = sorted(range(len(pairs)), key=lambda i: -lengths[i])
        results: list[LoglikelihoodResult | None] = [None] * len(pairs)
        pad = self.tokenizer.pad_token_id

        for idx in self._batches(order, lengths):
            seqs = [seqs_all[i] for i in idx]
            width = max(len(s) for s in seqs)
            input_ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs], device=self.model.device)
            attn = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=self.model.device)
            keep = max(len(pairs[i][1]) for i in idx)

            with torch.inference_mode():
                logprobs = torch.log_softmax(self._continuation_logits(input_ids, attn, keep).float(), dim=-1)

            for row, i in enumerate(idx):
                cont = pairs[i][1]
                n = len(cont)
                # Logits at the last n positions predict the n continuation tokens.
                lp = logprobs[row, keep - n :]
                target = torch.tensor(cont, device=lp.device)
                token_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                greedy = bool((lp.argmax(-1) == target).all())
                results[i] = LoglikelihoodResult(float(token_lp.sum()), n, greedy)
        return results  # type: ignore[return-value]
