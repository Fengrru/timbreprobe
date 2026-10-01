# Demo audio — hear the parameter space

16 anchor patches from `timbreprobe/corpus.py`, rendered by the bundled synthesizer
(0.5 s sustained C4 at 16 kHz, 16-bit PCM, loudness-normalised) and written by

```bash
python -m timbreprobe.cli corpus --preset smoke   # -> out/corpus_smoke/wav_anchors/
```

They are the corners of the corpus: bright lead, warm pad, dark sub, hollow
square, narrow pulse, FM brass, noise breath, pluck, resonant sweep, soft
flute, triangle organ, HP air, octave stack, sub-heavy pad, driven lead,
metallic FM.  Listening to them is the fastest way to calibrate what the
embedding's distance actually means.

> These are synthetic renders from the Stage-0 oracle synth (STFT-domain
> filtering), not recordings and not the target instrument.
