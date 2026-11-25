# Connectome Observatory

For current launch/install/resume commands, read [the agent runbook](../AGENTS.md).
This document includes historical viewer descriptions; the current game controls
are Pause/Step/Play under Game Status (see the final section). Freeze view is
removed, and All inputs includes visual, resource, and range populations.

## Visual inputs and control wiring

The top-right Game status area keeps the current scene, game frame, training
episode and action. The full-screen preview has been replaced by the bottom
strip of six **actual 42x42 encoder maps**: luminance, red-green, blue-yellow,
recent motion, four-frame motion and edges, plus encoded health/ammunition bars.
Positive/negative motion uses mint/violet. Each map arrives as float32 telemetry
from the same inference sample as neuron activity and action; maps are never
reconstructed from a later frame or a grayscale preview. Older producers without
map telemetry show a waiting message instead.

Thin lines connect these maps through the fixed input mapper to representative
sensory neurons. Click a map to filter its neuron links. Shift-click a map pixel to trace **all** of its assigned neurons;
the status line reports pixel coordinates, drive and destination count. All maps
restores the overview, and Mapping links hides/shows this overlay.
Mapping links and Readout traces both start off on every page load. Enable each
switch independently; Show all neuron links controls the density of readout
traces when they are enabled.
The visual-input strip also has Show all neuron links (off by default). With
Mapping links enabled and All maps selected, it draws all 11,426 input-neuron
links, including the 192 resource-input neurons. A selected map/pixel narrows
the displayed links to its actual destinations.

On the output side, neurons feed the nonlinear 256→128→36 readout box, which displays
the selected ID among 36 action combinations. Lines then lead to the controls
decoded from that action. By default, eight output-neuron links are shown,
ranked by activity magnitude; Show all neuron links displays the whole readout
population. These show structural wiring and activity, **not additive weight
contributions or causal attribution** for the nonlinear policy. Pin a combination
in the side panel to inspect it; illuminated control buttons still show the live
action. Inspection never sends game input.

Drag with the middle mouse button (press the wheel), or Shift-drag, to pan the
entire 3D scene. Ordinary left drag rotates; the wheel still zooms. Panning does
not pick neurons. Input-map telemetry was added at the completed episode 108 /
update 27 boundary, with the same readout, optimizer and learning configuration.

## Translucent fly body

The **Fly body** checkbox adds a faint mint/blue NeuroMechFly v2 exoskeleton
around the connectome, using the same 3D orbit, pan and zoom. **Fit** frames the
whole fly when enabled; **Brain** focuses on the neurons. The body starts off
on each page load; enable the checkbox to show it. The body does not participate in neuron picking and
does not affect inference or training.

The detailed mesh has 69 posed segments and 115,158 triangles, including the
eyes, antennae, wings, abdomen and six legs. It is derived from the Apache-2.0
NeuroMechFly assets distributed by
[NeLy-EPFL/fly-svg-maker](https://github.com/NeLy-EPFL/fly-svg-maker), pinned to
commit `152506d3471646f009480c81f34aefbaec29a6e5`. Attribution, the license,
per-source SHA-256 hashes, pose verification and derived mesh checksum are in
`python/viewer/web/assets/fly`. Rebuild with
`.\.venv\Scripts\python.exe scripts\prepare_fly_body.py`.

The head is scaled and positioned to contain the displayed brain. These are
different specimens and coordinate systems: this is illustrative spatial
context, **not a measured anatomical registration**. No neuron coordinates
are changed.

When a producer is already running on 47652, `Start-Observatory.ps1` opens a
read-only viewer on 47653, so updated display code can be used without
restarting training. The observer exits after 120 seconds with no viewer
heartbeat; opening the launcher again starts it if necessary.

The standalone Edge app window shows **all 139,255 identified FlyWire neurons**. A separate PyTorch process reads the actual 84x84 game observation, updates the sparse recurrent state and chooses one of 36 ship actions. The graph colors, inspector, episode, game frame and chosen action come from the **same inference sample**. The display is not an animation triggered by symbolic game events. Action is the policy's choice from that frame, not a promise that all requested ticks ran before death.

## Run on Windows

Start the Steam game with NeonFly plugin 0.4 and enter `Level_1` (the game-over screen also works). **The viewer's producer takes control of the game**, including normal scene resets. Do not run another Gym client or training process against the same instance. This version monitors its bundled experimental producer, not arbitrary external model processes.

```powershell
.\.venv\Scripts\python.exe -m pip install -r python\requirements-viewer.txt
# For this machine / Python 3.13, if torch is not already installed:
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1+cu128
.\scripts\Start-Observatory.ps1
```

First run acquires and checks approximately 220 MB of compressed public tables, then builds about 250 MB of local indices under `.deps\flywire783`. No game assets or frames leave the machine. CUDA is the default; `-Device cpu` is an explicit alternative (not the benchmarked configuration). The viewer is served at `http://127.0.0.1:47652/`, with read-only routes, bounded clients and no remote game-control endpoint. One browser window is sufficient. Freeze holds the displayed graph, input and action **without pausing inference or gameplay**. When all viewer tabs/windows are closed, the service stops itself and its agent after 120 seconds without a heartbeat. For an immediate manual stop, run the server in a foreground console (`python -m python.viewer.serve`) and press Ctrl+C. Game remains open. If the game ends up outside Level_1 or the bridge fails, the producer reports an error rather than replaying ambiguous inputs.

## What is measured vs assumed

| Component | Implementation |
|---|---|
| Identity and topology | Public Codex FAFB v783 `neurons`, `coordinates`, `classification`, and **unthresholded** Buhmann-era `connections_no_threshold` tables. `prepare.py` pins Google Storage object generations and verifies the source MD5s. All 139,255 root IDs retained as exact integers (strings in the browser); no region, weak edge or isolated cell is removed. Rows sharing a pre/post pair are summed across neuropils. **15,091,983 directed pairs; 54,492,922 synapses** in these pinned tables. Do not compare this edge total with thresholded Codex overview counts. Raw compressed tables remain separate from the numerical weights. |
| Layout and lines | One WebGL point per cell at its first representative FlyWire XYZ coordinate; not a soma or morphology reconstruction. Drag to orbit, Shift-drag to pan, scroll to zoom, and click or look up a FlyWire ID for exact inbound (violet) and outbound (mint) links. Optional overview bundles join 3D centroids of the original XY bins; bundles are off by default. Reset map clears the selection and restores the camera. |
| Role overlays | Blue rings mark the 11,426 neurons receiving pixels; orange rings mark the 1,415 neurons feeding the policy/value heads. Toggle each population independently. The center color continues to encode live activation; these roles describe model wiring, not inferred biological actions. |
| Ship control wiring | The actor scores 36 combinations, not six independent buttons. The side rail lights the six controls decoded from the current choice; neutral lights none. Follow the live action or pin any combination to inspect its score. Lines run from readout neurons to the numbered score, then to the buttons it implies. By default the eight largest absolute contributions are shown; enable all links or inspect a specific readout neuron. A contribution is the actor weight times that frame's feature (layer-normalized for PPO); bias is shown separately. Positive terms raise the score, negative terms lower it. Training samples actions rather than necessarily choosing the highest score. Inspection never sends game input. |
| Sensory encoding | Artificial 2D projection of the screen onto the representative coordinates of 11,426 cells classified as visual sensory. This is not a validated retinotopic map. Only the newest grayscale observation drives input; no game coordinates, reward or privileged state enters the model. |
| Dynamics | One float32 state per neuron, including isolated cells. Sparse incoming synapse counts are normalized by the postsynaptic cell's total count. Predicted GABA cells are assigned negative polarity; all other/unknown types are treated as positive. This simplification is **not measured synaptic sign**. Three leaky tanh recurrent updates per decision; state clears at episode reset. No fabricated edges in the recurrent graph. |
| Readout and learning | A reproducibly seeded, artificial random readout of 1,415 descending/motor cells selects the maximum-scoring action. **No RL optimization or trained checkpoint**. No claim of useful performance or biological fidelity of this policy. |

To display updated viewer code alongside a trainer already serving port 47652, run `.\.venv\Scripts\python.exe -m python.viewer.serve --observe --port 47653` and open `http://127.0.0.1:47653/`. This read-only process forwards telemetry without launching a second controller or modifying the checkpoint. It exits after two minutes without viewer requests.

## Validation and performance

For an uncapped PPO run, use `python -m python.viewer.serve --train-run RUN --resume --continuous`. This bypasses only the update limit, not checkpoint checks or evaluation. To hand off from an already running capped Windows trainer without losing its current batch, run `.\scripts\Continue-Training.ps1 -Run RUN -TrainerProcessId PID` with the PID listening on 47652. It holds a temporary idle-sleep inhibition and resumes only after the original trainer exits at its final checkpoint and evaluation. Creating `logs\ppo\RUN\STOP` cancels a waiting handoff and stops the trainer at an episode boundary. `latest.pt` is the last completed update, not a partial batch. The system still requires power, a working game, and disk space; there is no reboot recovery.

`python -m python.tests.verify_viewer_graph` independently streams the raw source table, verifies the full identity roster and total synapse count, compares both sparse indices against the exact partners and weights of 32 seeded neurons (2,935 pairs in this release), and checks that overview counts sum to the full directed-pair count.

On the Windows RTX 4070, a 12-second concurrent measurement captured 100 distinct snapshots over four real game episodes: **8.29 game decisions/s**, **5.24 ms median / 11.31 ms p95 inference** (includes copying the activity vector for telemetry), zero producer queue drops. These figures are one local observation, not a training benchmark or display FPS guarantee. The Unity PNG bridge and action durations still limit the game loop. Browser inspection confirmed successful WebGL rendering, an exact-ID lookup, advancing samples and freeze/resume.

Topology is transferred to WebGL once (positions and bundles); live samples send only metadata, an 84x84 grayscale plane and 139,255 float32 activations. The producer publishes without waiting on render or a client: the process queue holds at most two immutable snapshots, the HTTP process keeps one current packet, and the browser polls at up to 10 Hz with one in-flight request. Missed display frames are not queued indefinitely. Selected-neuron history is the observed display stream, not every recurrent substep.

**Attribution:** FlyWire Consortium; [Dorkenwald et al., Nature 2024](https://doi.org/10.1038/s41586-024-07558-y); [Codex credits and citation guidance](https://codex.flywire.ai/about_flywire); [FlyWire terms](https://flywire.ai/tos). Coordinate, class, neurotransmitter and synapse annotations have distinct contributors. Local raw and derived data are ignored by Git; review the applicable terms, attribution and third-party notices before redistributing them. The viewer uses system Edge/WebGL and the project-local PyTorch, NumPy, pandas (BSD), SciPy (BSD) and Pillow installations.

Health, Bullets, and Rockets are selectable input panels, each scoped to its actual 64 assigned neurons. The range sensor panel selects all 2,674 assigned mechanosensory neurons. Enable Mapping links to display pathways; Show all neuron links expands the representative subset to the full selected population. Resource link colors represent injected drive; range link colors represent measured neuron activity, not reconstructed sensor drive or causal contribution. All inputs restores the visual maps, all three resources, and the range sensor overview (when the live encoder provides a valid sensor). With Show all neuron links enabled, this displays 14,100 input links: 11,234 visual, 192 resource, and 2,674 range links. Panel selection also supports keyboard activation. Traces remain off by default. These observer-only changes do not alter the running policy or training inputs.

## Game advancement controls

Game Status contains Pause, Step, and Play. These control the running model
worker, not just the browser stream. Pause takes effect at an action boundary;
an action already in flight finishes first. Step, enabled only while paused,
executes exactly one model-selected action (currently eight physics ticks),
then holds again. Play resumes continuous training. The viewer keeps receiving
telemetry while paused; Freeze view/Resume view have been removed.

The paused worker renews the Unity bridge's hold once per second, so its
30-second abandoned-controller watchdog does not release a deliberate pause.
The game remains paused even when the browser closes; Play can be issued from
another viewer. If the worker disappears, the bridge watchdog still releases
the abandoned hold. A terminal step displays the final game state and waits
before the next episode reset. A later Step may reset the episode, then execute
one action. Terminal brain activity is the last decision's activity.

POST /control accepts only pause, step, or play, with same-origin and session
token checks. The observer on 47653 forwards these commands to the sole
controller on 47652; it never drives Unity independently. Read-only GET routes
remain read-only. The control helper and model-worker changes have an audited
checkpoint compatibility migration; learning settings and saved learning state
are preserved. The pre-upgrade checkpoint is retained as
checkpoints/flywire-sensor-005/before-viewer-controls.pt, and source snapshots
are in logs/ppo/flywire-sensor-005/source-controls.

Control validation: 30 Python tests pass, including exact one-action step
credits and terminal health handling. Live Pause held physics time, fixed-update
count, position, health and ammunition unchanged for 32 seconds. Live Step
increased completed actions from 71 to 72 and returned to Paused; Play resumed
advancing actions. Missing/foreign Origin POSTs were rejected. The retired
Freeze view button is absent. Evidence: logs/game-controls-verification.json
and logs/game-advancement-controls.png.
