# Contributing to LitheVoice

Thanks for considering it. Two rules keep this project healthy.

## 1. Sign your work (DCO)

Every commit must carry a Developer Certificate of Origin sign-off, certifying
you have the right to submit the code under Apache-2.0:

```
git commit -s
```

which adds `Signed-off-by: Your Name <you@example.com>`. The DCO text is at
<https://developercertificate.org>. No CLA, no paperwork — just the sign-off.

## 2. Measurements over opinions

This is a latency project. "Feels faster" is not a claim; a number from a
repeatable harness is. If a change is about speed, say what you measured, on
what hardware, and how someone else reproduces it — `tests/bench_tts.py` and
`tests/bargein_sim.py` exist for exactly this, and the simulation touches no
audio hardware, so it is repeatable on any machine.

If a change is about correctness, bring a test that **fails before it and
passes after**. `tests/test_lite.py` is the model to follow: its docstring
records the bug that motivated it — an ONNX VAD wrapper that returned a
near-zero probability for everything, so the microphone was simply deaf and
nothing raised. Tests that only check "stays quiet on silence" passed happily
against it. The useful check was real speech, frame by frame, against the
reference implementation.

## Running the tests

```bash
python -m pytest tests/test_release.py tests/test_webui.py -q   # no models needed
./.venv/bin/python -m pytest tests/ -q                          # full, after setup
```

Tests skip rather than fail when an optional backend is missing, so a red
suite means a real regression. Please keep it that way.

## Model pinning

`scripts/models.json` pins every downloaded artifact by revision and SHA-256,
and `scripts/download_models.py` verifies both before a file is accepted. If
you change a pin, update the hash in the same commit and say why it moved.
