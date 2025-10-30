from __future__ import annotations

from pathlib import Path
import torch


def save_probe_weights_zero2(model, out_path: Path) -> None:
    """Save probe weights, gathering under ZeRO-2 if available.

    Works whether or not DeepSpeed is present; falls back to direct state_dict.
    """
    try:
        probe_sd: dict[str, torch.Tensor]
        zero_ctx = None
        try:
            from deepspeed import zero as ds_zero  # type: ignore
            zero_ctx = ds_zero.GatheredParameters(list(model.probe.parameters()), modifier_rank=0)
        except Exception:
            try:
                from deepspeed.runtime.zero import GatheredParameters as _GP  # type: ignore
                zero_ctx = _GP(list(model.probe.parameters()), modifier_rank=0)
            except Exception:
                zero_ctx = None
        if zero_ctx is not None:
            with zero_ctx:
                probe_sd = {k: v.detach().cpu() for k, v in model.probe.state_dict().items()}
        else:
            probe_sd = {k: v.detach().cpu() for k, v in model.probe.state_dict().items()}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(probe_sd, out_path)
    except Exception as e:
        print(f"Warning: failed to save probe weights cleanly ({e})")


