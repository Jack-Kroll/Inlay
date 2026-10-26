# Pretrained audio model benchmark — 2026-10-19

Compared four released checkpoints with the current Inlay pipeline. No model was trained, no acceptance threshold was tuned, and no runtime default was changed. The original 90.21% GuitarSet / 85.03% IDMT note F1 baseline reproduced exactly.

**Result:** none is a reliable drop-in replacement across these two datasets. YourMT3 Single and MoE reach 94.30% and 93.62% note F1 on GS05, but only 72.14% and 72.26% on IDMT, versus current Inlay at 85.03%. The most useful measured component is the public GuitarSet-finetuned SynthTab string posterior: on fixed current notes, IDMT tab F1 rises from 49.71% to 52.80%. This is useful evidence, but it falls short of the requested breakthrough.

**Next experiment:** keep the current event detector as the baseline and test the stronger string evidence jointly with visible fret/finger contacts, including the open-G versus fret-10 failure. New note candidates from YourMT3 could supply additional proposals, but selecting or verifying them with synchronized vision remains an untested hypothesis. Evaluate that on independent electric-guitar video with reference tabs before promoting a model or fusion policy. Another unrestricted audio-model swap is not supported by these results.

## Evaluation

- All 60 GuitarSet player-05 microphone excerpts: 8,715 annotated notes.
- All 657 locally available standard-tuning IDMT recordings: 5,164 annotated notes.
- Six native clips: Blackbird, Loser, Seven Nation Army, Sweet Child O Mine, take-4, Thunderstruck.
- Primary metric: micro note F1 with one-to-one pitch/onset matching, 50 ms onset tolerance, 50-cent pitch tolerance, offsets ignored.
- Tab F1 additionally requires the correct string using the same note matching. The audio-only tests provide no video frames.
- All non-drum YourMT3 predictions are scored. A predeclared guitar-program-only diagnostic is reported separately.
- Predictions, references, input/model hashes, source snapshots, per-file timing, and setup versions are retained in the benchmark bundle.

**Training overlap:** player 05 is held out for Inlay, but upstream YourMT3 uses other GuitarSet splits, and SynthTab fine-tuning shuffles recordings across players. Full GS05 comparisons can include training overlap. They are not independent unseen-player claims. The progression-3 slice is retained as a diagnostic, not asserted to be a fresh common holdout. IDMT has been repeatedly inspected in earlier campaigns; it remains measurement-only, and these results are regression/model-screening evidence. No IDMT-finetuned checkpoint is included.

## Native event output

| Model | GS05 note F1 | IDMT note F1 | GS05 tab F1 | IDMT tab F1 |
|---|---:|---:|---:|---:|
| Current Inlay v8 | 90.21% | 85.03% | 78.44% | 49.71% |
| YourMT3+ Single (noPS) | 94.30% | 72.14% | — | — |
| YourMT3+ MoE Multi (noPS) | 93.62% | 72.26% | — | — |
| SynthTab TabCNN x4, synthetic only | 52.02% | 48.23% | 29.23% | 24.62% |
| SynthTab TabCNN x4, public GuitarSet fine-tune | 78.15% | 48.31% | 75.72% | 32.02% |

YourMT3 has no string/fret output, so a tab F1 is not fabricated for it. SynthTab predicts framewise string/fret states; its event scores use contiguous same-string/same-fret runs with no duration threshold or onset adjustment. This decoder cannot distinguish unchanged-fret replucks without an intervening state change. Those event scores describe this adapter, not the frame metric published by SynthTab.

| Model | IDMT precision | IDMT recall |
|---|---:|---:|
| Current Inlay v8 | 86.44% | 83.68% |
| YourMT3+ Single (noPS) | 72.76% | 71.53% |
| YourMT3+ MoE Multi (noPS) | 80.81% | 65.34% |
| SynthTab TabCNN x4, synthetic only | 35.71% | 74.24% |
| SynthTab TabCNN x4, public GuitarSet fine-tune | 35.14% | 77.29% |

## Framewise diagnostic

Frame pitch F1 compares active rounded-MIDI pitch sets on the common 512/22050-second grid. Frame tab F1 additionally requires the same string and fret. These sustain/frame metrics are different from pitch/onset note F1.

| Model | GS05 frame pitch F1 | IDMT frame pitch F1 | GS05 frame tab F1 | IDMT frame tab F1 |
|---|---:|---:|---:|---:|
| Current Inlay v8 | 84.81% | 86.10% | 75.25% | 55.95% |
| YourMT3+ Single (noPS) | 91.50% | 70.69% | — | — |
| YourMT3+ MoE Multi (noPS) | 90.50% | 68.78% | — | — |
| SynthTab TabCNN x4, synthetic only | 71.55% | 78.95% | 46.47% | 47.62% |
| SynthTab TabCNN x4, public GuitarSet fine-tune | 93.10% | 84.34% | 91.65% | 59.33% |

## Same-note string evidence

Every Inlay decoded pitch and onset is held fixed. At each current onset, the nearest SynthTab frame is softmaxed over the 21 fret/silence classes on each string. Probabilities at physically compatible string/fret spellings are normalized across strings. Only the string posterior changes; the existing .25 audio weight, open-neutral 1, chord solver, and neck-position planner remain fixed. Numerically empty or wholly out-of-range posteriors retain the original estimate. No timing offset, threshold, or fusion weight is fitted.

| String evidence | GS05 tab F1 | IDMT tab F1 | GS05 string accuracy on matched notes | IDMT string accuracy on matched notes |
|---|---:|---:|---:|---:|
| Current Inlay v8 | 78.44% | 49.71% | 86.95% | 58.46% |
| SynthTab TabCNN x4, synthetic only | 60.82% | 46.32% | 67.42% | 54.48% |
| SynthTab TabCNN x4, public GuitarSet fine-tune | 83.85% | 52.80% | 92.95% | 62.09% |

All variants preserve the baseline note F1 exactly: 90.21% / 85.03%. The public GuitarSet-finetuned string model raises fused tab F1 from 78.44% to 83.85% on GS05 and from 49.71% to 52.80% on IDMT. The GS05 improvement has training-overlap caveats. This is a modest component gain, not a new high-accuracy end-to-end system.

Bypassing fusion and using the compatible-string argmax directly gives the original head 89.76% / 60.59% string accuracy on matched GS05 / IDMT notes, versus 97.42% / 65.31% for the GuitarSet-finetuned SynthTab head. Those conditional scores are not complete tab accuracy, and direct argmax has not been validated as a runtime policy.

## Native footage

All four new checkpoints inferred audio from all six native clips. There are no independent reference tabs for these clips, so counts and support labels are not scored accuracy.

The YourMT3+ Single prototype uses the original cached vision with its new note events. YourMT3 has no audio string posterior, so this replay uses geometry/context alone. It is not a controlled test of pitch changes alone. On Seven Nation Army, the G3 at 1.66 s lands on string 5, fret 10; the 5.94 s G3 still lands on open string 3. The earlier correction is consistent with removing the old audio-string influence; it does not establish that a stronger pitch detector solved the placement policy.

[Seven Nation Army experimental replay](</Users/jackkroll/CS-Projects/inlay/runs/audio-model-benchmark-2026-10-19/native-replay/yourmt3-single-nops/seven-nation-army/Seven Nation Army.overlay.mp4>)

The rendered Seven Nation Army file has 518 decoded video frames, a 1620×1080 video stream, an audio stream, and duration 17.268394 s. The checked frames show both the corrected first placement and the remaining second error. Normal transcription folders and their current defaults remain unchanged.

MoE was also replayed through the same cached vision on all six clips. Its Thunderstruck replay emits 133 notes, compared with 95 in the current canonical transcription and 96 in the Single replay; 37 MoE assignments receive the unsupported `none` label. This is a qualitative warning, not a false-positive count against ground truth. The rendered file has 464 decoded 1620×1080 frames, audio, and duration 15.468213 s; a checked frame still shows an unsupported open-string placement.

[Thunderstruck MoE experimental replay](</Users/jackkroll/CS-Projects/inlay/runs/audio-model-benchmark-2026-10-19/native-replay/yourmt3-moe-multi-nops/thunderstruck/Thunderstruck.overlay.mp4>)

## Hardware and timing

Apple M5, 16 GiB shared memory, 10 CPU cores. Alternate-model environment: Python 3.11, PyTorch 2.5.1, NumPy 1.26.4, Transformers 4.45.1. Full-precision weights and inference were retained. CPU workers use four threads. GPU inference used Metal/MPS with CPU audio preparation and symbolic decoding.

Single ran on MPS with batches 16 then 8. MoE began on MPS batch 8 and completed on CPU batch 4 after profiling showed the CPU route faster on this hardware. Both SynthTab checkpoints ran on MPS. On the profiled 32.26-second recording, MoE took about 12 s on CPU, 24 s on MPS batch 8, and 42 s on MPS batch 1, with matching note-match/prediction/reference counts. FP16 did not improve the repeated warm pilot; the attempted BF16 autocast was disabled by this runtime.

| Model | Recorded successful inference time, labelled audio | Audio duration | Devices |
|---|---:|---:|---|
| Current Inlay v8 | Cached baseline; unavailable | 105.55 min | Original cached run |
| YourMT3+ Single (noPS) | 12.46 min | 105.55 min | mps |
| YourMT3+ MoE Multi (noPS) | 36.85 min | 105.55 min | cpu, mps |
| SynthTab TabCNN x4, synthetic only | 1.99 min | 105.55 min | mps |
| SynthTab TabCNN x4, public GuitarSet fine-tune | 1.94 min | 105.55 min | mps |

Times include audio loading/resampling, feature extraction and inference/postprocessing, but exclude checkpoint loading, result-file writes, profiling, interrupted calls, scoring and video rendering. They mix cold/warm calls and were measured on a shared machine; they are practical observations, not an idle-machine speed benchmark. No baseline speedup ratio is claimed from replaying its cache.

## Sources and reproduction

- [Official YourMT3+ release](https://github.com/mimbres/YourMT3) and [checkpoint model card](https://huggingface.co/mimbres/YourMT3).
- [Official SynthTab source and checkpoints](https://github.com/yongyizang/SynthTab/tree/main/demo_embedding).
- Exact downloaded weights, safely extracted tensor weights, source-file SHA-256 values, source revisions, input fingerprints, immutable protocol/amendments, per-recording predictions, and the Inlay source snapshot are in the bundle.
- The alternate environment is isolated from Inlay’s application environment; see `environment-freeze.txt` and the bundle `README.md`.
- Checkpoint loading for inference uses `weights_only=True`. Official serialized objects were converted to tensor dictionaries with the restricted loader and inert local placeholders; external checkpoint classes were not executed during conversion. All inference state-dictionary keys matched.
- Ten benchmark checks cover onset/pitch matching, one-to-one unisons, missing string assignments, empty outputs, frame-vs-tab scoring, upstream TabCNN patch layout, and string numbering/fallbacks.
- Upstream source licenses and attribution remain with the copied sources. IDMT remains measurement-only; no data or checkpoint was trained in this campaign.
