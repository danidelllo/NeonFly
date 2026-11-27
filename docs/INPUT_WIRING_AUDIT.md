# Input wiring audit — trainable brain 016, 2026-11-27

Read-only audit; no remapping, parameter changes, restarts or game actions.

## Verified data path

| Raw vector columns | Meaning | Assigned cells |
| --- | --- | ---: |
| 0–881 | 126 rays x 7 channels, interleaved per ray | 2,646 |
| 882–885 | vx / 8, vy / 8, normalized viewport x, y | 12 |
| 886 | normalized health fraction | 64 |
| 887 | normalized bullet fraction | 64 |
| 888 | normalized rocket fraction | 64 |

Each ray's channel order is proximity, closing speed, solid proximity, target
proximity, hostile-shot proximity, lethal proximity, visible-range fraction.
Radar/body values have exactly three distinct neuron destinations per value;
resources have 64 each. Total: 889 values, 2,850 unique assigned neurons, no
overlapping populations or out-of-bounds assignments. Resource neuron blocks
and signal order match Unity fields, Python packing, injection and viewer roles.

Tests assign every raw channel a distinct nonzero value and measure the actual
neural transition with recurrence isolated: no drops or slot offsets. An
independent per-ray reference verifies all category, closing-speed, body and
resource encodings. Online stepping, complete sequence replay and chunked
sequence replay produce exactly identical forward states for the same inputs.

Sparse graph rows are postsynaptic destinations and columns presynaptic sources.
All stored matrix values exactly matched an independently calculated incoming
normalization from raw synapse counts, with the configured source-neuron sign
convention. These engineering conventions are not claims about measured fly
physiology. Visual drive remains disabled.

## Live checks

The active viewer mapping matched the model's neuron and slot arrays exactly.
Independent linear logits reconstructed from the published full-brain activity
and matching actor weights differed from the live GPU scores by at most
9.54e-7 in the sampled packet. Only the first 36 linear outputs select actions;
the value unit is training-only. No hidden decoder layer.

A saved real raw-sensor episode had correct shape and finite bounded values;
category channels summed to generic proximity with no multiple active hit
categories. Closing speed reconstructed from adjacent recorded proximities and
the controlled 0.16-second action interval differed by at most 7.75e-7. First
decision closing speeds were zero, confirming reset of previous-scan context.

## Structural caveat, not a dropped channel

Reverse graph traversal from all 1,415 motor/descending readouts found a path for
2,715 of 2,850 assigned input cells. 135 assigned cells have no path to these
readouts in the supplied connectome. Redundant mapping means ALL 889 signals
still have at least one connected destination. For the 886 radar/body channels:
787 have three reachable copies, 97 have two, and two have one. Health, bullets
and rockets have 54, 50 and 54 reachable assigned cells respectively.

No input channel is structurally absent, but not every nominal copy can affect
the selected readouts. Anatomical reachability does not prove that a trained
network can discriminate every situation. Do not change mappings mid-experiment
or call this audit evidence of gameplay success.

## Regression checks

71 Python tests passed, including four new input-wiring checks. Compiled C# tests
passed 10,012 camera-clipping cases and all 126 sector-angle checks. Commands:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s python/tests
.\scripts\Test-RangeClip.ps1
```

Live numeric evidence is saved locally in
logs/ppo/flywire-plastic-016/wiring-audit.json. Training continued uninterrupted.
