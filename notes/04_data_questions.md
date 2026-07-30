# Open questions about the data

Status: **the Dropbox data is not accessible yet** — the local folder has only the three PDFs
and `1D_HD_plots.docx`. Everything below is inferred from JP's email and from Eric's figures.

## Run this first, before any analysis

```python
from seven_a.matio import load_mat, describe
data = load_mat("data/<file>.mat")
print(describe(data, max_depth=7))
```

`describe()` prints unique values for small numeric arrays, which is how the `choice` coding
question gets settled in one look rather than by argument.

---

## Questions that change which analyses are possible

### 1. Are there cue-conflict (Δ) trials?  **Blocks: behavioural cue weights (Gu 2008)**
Gu 2008 used Δ = ±4°: visual displaced +Δ/2, vestibular −Δ/2. Without conflict trials we can
still test the *threshold* prediction (√ combination) but cannot measure the *weight* the animal
placed on each cue, which is half of that paper's argument.
Look for a field named `delta`, `conflict`, or a second heading vector per trial.

### 2. Is visual coherence varied?  **Blocks: the entire Fetsch 2011 replication**
Fetsch's result *is* the reliability manipulation — behavioural and neuronal weights shifting with
coherence. If every trial is at one coherence, template paper C is not reproducible and the paper
rests on A + B. Worth knowing now rather than after building toward it.
Look for `coherence`, `coh`, `motion_coherence`.

### 3. What exactly is in `choice`?
JP: values are 0 and 5, or −5 and +5, and the meaning appears to flip with heading sign.
`resolve_choice_coding()` handles this automatically, but two things need confirming:
- **Are there more than two distinct values?** A third value (aborts, no-choice) must be filtered
  first; the resolver raises rather than guessing.
- **Zero-heading trials.** If the coding really is stimulus-relative, then at 0° there is no
  correct answer and the stored value cannot be decoded — the information is gone. Those trials
  are currently dropped. That matters because **0° is exactly where choice probability is most
  informative** (no stimulus drive to remove). If there is a separate field recording the
  rewarded side or the actual choice target for 0° trials, it rescues the most valuable trials in
  the dataset. Worth asking JP directly.

### 4. Is spiking stored as counts, rates, or spike times?
Determines whether we can choose the integration window ourselves. Preferably spike times, so we
can (a) match Gu's analysis window, (b) compute CP time courses, and (c) recompute rates over the
middle stimulus period rather than inheriting someone's choice of window.

### 5. Are units linked to sessions, channels, and monkeys?
Needed for three things:
- **Population decoding** (Fig 6) requires knowing which units were recorded *simultaneously*.
  The 16-channel U-Probe makes this possible and it is the strongest available answer to 7a's poor
  single-neuron sensitivity — but only if session identity is recoverable.
- Reporting n per monkey, as the template papers do.
- Not treating repeated recordings of the same unit as independent.

### 6. Is there a multi-unit structure alongside `singleunits`?
Eric's doc counts 146 single + 256 multi units for HD. MUA roughly triples the yield and, given
how weak single-unit modulation is, may be where the population signal actually lives.

### 7. What is the relationship between the HD sessions and the bhy272 dataset?
bhy272 is forward translation at varying *speed*, 700 ms, passive. The 1D Azi plots show 8
azimuths, 2 s, passive. HD is ±12° heading discrimination, active. These look like three distinct
protocols. Are the HD recordings from the same animals/sessions as bhy272, or a separate set?
Affects whether we can cite Avila 2019's tuning characterisation as applying to these units.

---

## Things to verify in the data itself, not by asking

| Check | Why |
|---|---|
| Trial counts per heading × modality | Determines whether per-session psychometric fits are stable, and whether CP has enough trials per condition (need ≥3 of each choice per heading) |
| Firing-rate distribution | Eric's plots suggest 1–5 spk/s. If true, single-neuron CP will be badly underpowered and the population approach is mandatory, not optional |
| Whether combined neurometric thresholds really are *worse* than single-cue | Eric's example session shows ves 13.65, vis 20.37, com **35.59**. Either a real effect carried by opposite cells (interesting), or an artefact (must be caught early) — see `02_eric_2018_plots_inventory.md` |
| Whether the exact 4.05/3.50/2.65 optimality in Eric's example replicates | Too clean for one session. Check it is a fit and not a back-computed value |
| Lapse rates | Unmodelled lapses inflate threshold estimates and would corrupt the optimality test |
