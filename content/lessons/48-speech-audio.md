---
title: "Speech & Audio AI"
stage: 5
order: 48
minutes: 45
difficulty: intermediate
prerequisites: ["transformers"]
tags: ["audio", "speech", "whisper"]
summary: "How machines hear — turning sound into spectrograms, speech-to-text with Whisper, and text-to-speech, from waveform to model input."
---

# Speech & Audio AI

## Why this matters

Every model this roadmap has covered so far eats text or images. Sound is a
different kind of signal — it unfolds over time, and "meaning" lives in
frequencies that aren't visible if you just look at the raw numbers. Once you
can turn sound into something a Transformer can read, the same architecture
from two lessons ago — attention over a sequence — handles audio too:
Whisper, the model behind most speech-to-text you've used, is a Transformer
underneath. This lesson is about that conversion step, since it's the part
that's genuinely specific to audio.

## The concept

**Sound as a waveform.** A microphone measures air pressure many times per
second and stores each measurement as a number — **amplitude**, how loud the
signal is at that instant. String those numbers together in time order and
you have a **waveform**: one long sequence of amplitudes. The **sample
rate** is how many measurements are taken per second, in **Hz** (hertz).
CD-quality audio samples at 44,100 Hz; speech models often use 16,000 Hz,
since human speech doesn't need the extra resolution music does. One second
of 16 kHz audio is 16,000 numbers before anything about *what was said* has
been extracted at all.

**Why models don't eat raw waveforms.** A waveform tells you loudness at
each instant, but "what pitch" or "what phoneme" isn't a property of any
single number in that sequence — it's a property of the *pattern* the
numbers make over a short stretch of time. A one-second clip at 16 kHz is
16,000 timesteps; feeding that directly into an attention mechanism, where
every position attends to every other position, is expensive and the raw
amplitude values don't line up well with what carries meaning. What you
want instead is a representation of frequency content — which pitches are
present and how loud each one is, at each point in time — and audio models
compute exactly that before doing anything else.

**The Fourier transform and the spectrogram.** The **Fourier transform** is
the mathematical tool that takes a signal in time and decomposes it into
the frequencies that make it up: any waveform, however complicated, can be
described as a sum of sine waves at different frequencies and amplitudes,
and the Fourier transform recovers exactly which ones and how strong. Run
it on a short window of audio (a few milliseconds) instead of the whole
clip, slide that window forward, and repeat — that's the **Short-Time
Fourier Transform (STFT)** — and you get a 2D picture: frequency on one
axis, time on the other, and brightness showing how much energy is at each
frequency at each moment. That picture is a **spectrogram**. A **mel-spectrogram**
applies one more step: it reweights the frequency axis to match how human
hearing perceives pitch (we're far more sensitive to differences at low
frequencies than high ones), which is the specific representation most
speech models are trained on. The "In code" section below runs a plain FFT
(Fast Fourier Transform, the standard efficient algorithm for computing a
Fourier transform) on a synthetic signal — the single-window version of
what an STFT does repeatedly to build a spectrogram.

**Automatic speech recognition (ASR) and Whisper.** ASR is the task of
turning speech audio into text. **Whisper**, from OpenAI, is a Transformer
trained on this task: audio goes in as a (mel-)spectrogram, gets encoded by
a Transformer encoder into a sequence of vectors — exactly the encoder
described two lessons ago — and a Transformer decoder generates the
transcript token by token, attending back to that encoded audio through
cross-attention, the same mechanism a translation decoder uses to attend
back to its source sentence. It was trained on hundreds of thousands of
hours of multilingual audio paired with transcripts, which is why it
generalizes across accents, background noise, and languages far better than
older ASR systems built for one narrow domain.

**Text-to-speech (TTS): the other direction.** TTS models go from text to a
waveform — the reverse problem. A common structure is two stages: a model
predicts a mel-spectrogram from text (learning which sounds and prosody the
words imply), and a second model — a **vocoder** — converts that
spectrogram back into an actual waveform, since a spectrogram alone
discards information (specifically, phase) needed to reconstruct exact
audio samples. Newer end-to-end models collapse this into fewer stages, but
the text-to-spectrogram-to-waveform pipeline is still the concept worth
having in your head.

**Audio classification.** Not every audio task involves language. Given a
clip, classify what's happening in it — a dog barking, a siren, a genre of
music, a speaker's identity — is audio classification, and it uses the same
spectrogram-as-input idea: convert the waveform to a spectrogram, then treat
that spectrogram like an image and run it through a classifier (a CNN or a
Transformer), the same way Stage 4's vision lessons classified pictures.

## In code

Real Whisper transcription needs an actual audio file and a model download,
so that part is illustrative below. What every audio pipeline needs first —
turning a waveform into frequency information — is fully runnable with
nothing but NumPy: `pip install numpy`

```python
import numpy as np

np.random.seed(0)

sample_rate = 8000          # samples per second
duration = 1.0               # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

freq1, freq2 = 220.0, 880.0   # Hz -- two pure tones mixed together
signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
noise = 0.3 * np.random.randn(len(t))
signal_noisy = signal + noise

print("First 10 samples of the noisy waveform:")
print(np.round(signal_noisy[:10], 3))

fft_result = np.fft.rfft(signal_noisy)
fft_freqs = np.fft.rfftfreq(len(signal_noisy), d=1 / sample_rate)
magnitude = np.abs(fft_result)

top_indices = np.argsort(magnitude)[::-1][:2]
top_freqs = sorted(fft_freqs[top_indices])

print("\nTop 2 frequencies recovered from the FFT (Hz):")
print(np.round(top_freqs, 1))
```

```
First 10 samples of the noisy waveform:
[0.529 0.611 1.124 1.606 1.382 0.313 0.724 0.394 0.609 1.092]

Top 2 frequencies recovered from the FFT (Hz):
[220. 880.]
```

Look at the first 10 raw samples: nothing about "220 Hz and 880 Hz" is
visible there, just noisy numbers bouncing around. The FFT recovers both
exact frequencies anyway, even with random noise added on top — this is
what a spectrogram exposes that a waveform hides, and it's the reason every
audio model converts to a frequency representation before doing anything
else.

Real transcription with Whisper (needs `pip install openai-whisper` and an
actual audio file — not runnable here without one):

```python
# ILLUSTRATIVE -- requires an audio file and the whisper package/model weights
import whisper

model = whisper.load_model("base")
result = model.transcribe("recording.wav")
print(result["text"])
```

Whisper's own preprocessing does the waveform-to-mel-spectrogram conversion
internally before its Transformer encoder ever runs — the FFT above is the
core of what that step computes, just simplified to one window instead of a
sliding one.

## Build this

Modify the code above: add a third sine wave at a frequency of your choice
(pick something clearly separated from 220 and 880 Hz, like 3000 Hz — but
keep it below `sample_rate / 2 = 4000` Hz, the highest frequency this sample
rate can represent), add it into `signal` before the noise, and change
`top_indices` to pull out the top 3 frequencies instead of 2. Confirm all
three of your chosen frequencies come back out.

**Stretch:** instead of running one FFT over the whole 1-second clip, split
`signal_noisy` into 10 non-overlapping windows of 0.1 seconds each, run
`np.fft.rfft` on each window separately, and print the dominant frequency
per window. That loop — FFT on a short window, slide forward, repeat — is
exactly the STFT that builds a spectrogram; stacking your 10 per-window FFT
outputs into one 2D array (windows x frequencies) *is* a (crude) spectrogram.

## Go deeper

- [OpenAI Whisper GitHub repository](https://github.com/openai/whisper) — the model, code, and paper describing how it was trained.
- [3Blue1Brown: But what is the Fourier Transform? A visual introduction](https://www.youtube.com/watch?v=spUNpyF58BY) — the clearest visual explanation of the exact idea this lesson's code demonstrates numerically.
- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course) — free, hands-on course covering spectrograms, ASR, TTS, and audio classification with real models.
- [Google Cloud: What is a Spectrogram?](https://cloud.google.com/blog/products/ai-machine-learning/spectrograms-what-are-they-and-why-do-we-need-them) — a short, practical explanation of spectrograms specifically.

**Next:** [Multimodal AI (CLIP & Vision-Language Models)](49-multimodal.md)
