from __future__ import annotations
import torch

if not getattr(torch.library, '_amp_safe_register_fake_installed', False):
    _orig_register_fake = torch.library.register_fake

    def _safe_register_fake(op_name, *args, **kwargs):
        decorator = _orig_register_fake(op_name, *args, **kwargs)

        def _wrapped(fn):
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if 'does not exist' in str(exc):
                    return fn
                raise

        return _wrapped

    torch.library.register_fake = _safe_register_fake
    torch.library._amp_safe_register_fake_installed = True
