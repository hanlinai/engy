"""Serve-side hidden-states trim — put this folder on the sglang serve's PYTHONPATH.

The proof needs the final hidden state of each generated token plus the one
prefill row that samples output token 0. Stock sglang returns the whole prefill
block, and re-attaches the full accumulated block on every stream step. Both
wraps below are in memory; nothing on disk is modified.

NEVER on a validator serve — the recompute reads the full prefill block.
Verified against sglang 0.5.12.post1.
"""
import sys


class _LastRowProxy:
    """Slice `t[a:b]` (one request's prompt rows) returns only the last row."""

    __slots__ = ("_t",)

    def __init__(self, t):
        self._t = t

    def __getitem__(self, k):
        if isinstance(k, slice) and k.step is None and k.stop is not None:
            # Clamp to the tensor end first: under chunked prefill it holds only
            # the last chunk's rows, fewer than prompt_len.
            stop = min(k.stop, self._t.shape[0])
            start = max(stop - 1, 0 if k.start is None else k.start)
            return self._t[start:stop]
        return self._t[k]

    def __getattr__(self, name):
        return getattr(self._t, name)

    def __len__(self):
        return len(self._t)


def _install_skip_prefill():
    from sglang.srt.managers import scheduler_output_processor_mixin as M

    mixin = M.SchedulerOutputProcessorMixin
    orig = mixin.process_batch_result_prefill

    def wrapped(self, *args, **kwargs):
        for a in list(args) + list(kwargs.values()):
            lo = getattr(a, "logits_output", None)
            hs = getattr(lo, "hidden_states", None) if lo is not None else None
            if hs is not None and not isinstance(hs, _LastRowProxy):
                lo.hidden_states = _LastRowProxy(hs)
        return orig(self, *args, **kwargs)

    mixin.process_batch_result_prefill = wrapped


def _install_only_last_hidden():
    from sglang.srt.managers import io_struct as IO

    # 0.5.12.post1 names it BatchTokenIDOutput; older trees BatchTokenIDOut.
    cls = getattr(IO, "BatchTokenIDOutput", None) or IO.BatchTokenIDOut
    orig_init = cls.__init__

    def wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        hs = getattr(self, "output_hidden_states", None)
        fr = getattr(self, "finished_reasons", None)
        # The receiver replaces meta_info["hidden_states"] on each recv, so only
        # the finish step's copy is ever read. Blank the rest — but only when the
        # list aligns 1:1 with the finish reasons, or slots cannot be matched.
        if hs and fr and len(hs) == len(fr):
            self.output_hidden_states = [
                h if f is not None else [] for h, f in zip(hs, fr)
            ]

    cls.__init__ = wrapped_init


try:
    _install_skip_prefill()
    _install_only_last_hidden()
    print("[engy] hidden-states trim armed (skip_prefill, only_last_hidden)",
          flush=True)
except Exception as e:  # never break interpreter startup
    print(f"[engy] hidden-states trim not armed: {e!r}", file=sys.stderr)
