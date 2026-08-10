#!/usr/bin/env python3
"""CPU-only regression checks for accumulation and mixup state handling."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer" / "classification" / "in1k_trainer.py"
SPEC = spec_from_file_location("mergenet_delivery_trainer", TRAINER)
trainer = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)


def collect(loader_len, update_freq):
    return [
        trainer._accumulation_step(index, loader_len, update_freq)
        for index in range(loader_len)
    ]


def main():
    assert collect(4, 1) == [
        (True, 1, 0, 4),
        (True, 1, 1, 4),
        (True, 1, 2, 4),
        (True, 1, 3, 4),
    ]
    assert collect(8, 4) == [
        (False, 4, 0, 2),
        (False, 4, 0, 2),
        (False, 4, 0, 2),
        (True, 4, 0, 2),
        (False, 4, 1, 2),
        (False, 4, 1, 2),
        (False, 4, 1, 2),
        (True, 4, 1, 2),
    ]
    assert collect(6, 4)[-2:] == [
        (False, 2, 1, 2),
        (True, 2, 1, 2),
    ]
    try:
        trainer._accumulation_step(0, 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("update_freq=0 must fail")

    enabled_mixup = SimpleNamespace(mixup_enabled=True)
    disabled_mixup = SimpleNamespace(mixup_enabled=False)
    prefetch_args = SimpleNamespace(prefetcher=True)
    host_args = SimpleNamespace(prefetcher=False)
    loader_on = SimpleNamespace(mixup_enabled=True)
    loader_off = SimpleNamespace(mixup_enabled=False)
    assert trainer._batch_mixup_is_active(enabled_mixup, host_args, loader_off)
    assert not trainer._batch_mixup_is_active(disabled_mixup, host_args, loader_off)
    assert trainer._batch_mixup_is_active(None, prefetch_args, loader_on)
    assert not trainer._batch_mixup_is_active(None, prefetch_args, loader_off)

    print("ACCUMULATION_AND_MIXUP_STATE_TEST_PASS")


if __name__ == "__main__":
    main()
