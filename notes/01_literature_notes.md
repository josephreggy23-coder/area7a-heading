# Literature notes — visuo-vestibular self-motion, area 7a

## 1. Avila, Lakshminarasimhan, DeAngelis & Angelaki (2019), *Cereb Cortex* 29:3932–3947 (`bhy272.pdf`)

The paper that produced the dataset lineage. **Passive only** — no choices.

**Protocol.** 3 male rhesus macaques, 16-ch linear array (U-Probe, 100 µm spacing) in 7a.
Two protocols per session:
- *Translation*: forward (straight-ahead) only. 4 peak speeds: 5.5, 11, 16.5, 22 cm/s.
  Conditions: vestibular / visual / combined.
- *Rotation*: yaw. 5 speeds × 2 directions (±15…±75 °/s). Unimodal only (combined was
  technically infeasible).

700 ms trapezoidal motion, ~500 ms flat phase used for rate estimates. Visual motion flanked by
400 ms of static dots. Fixation on head-fixed target throughout. ≥5 reps per condition, plus
fixation-only baseline trials.

**Key findings.**
- *Vestibular dominance*, contrary to the received view that 7a is "a visual area."
- Only a small fraction of neurons show visual/vestibular **or** linear/angular convergence →
  argues for largely **independent population codes**.
- Responses scale with speed (mostly monotonic), but temporal dynamics are diverse.
- Signal and noise correlations both fall off with electrode distance → spatial clustering by
  sensory preference. No systematic laminar/columnar structure otherwise.
- ~30% of single units responsive to radial optic flow (lower than earlier single-electrode
  studies, attributed to reduced sampling bias + restricted flow patterns).

**Methods worth reusing.**
- Responsiveness: 250 ms post-onset vs 250 ms pre-onset, two-sided t-test, p ≤ 0.05.
- Speed tuning: one-way ANOVA p ≤ 0.05.
- Speed Selectivity Index, `SSI = (r_max − r_min) / (r_max − r_min + 2·sqrt(SSE/(N−M)))`
  — a modulation index normalised by intrinsic variability. (Takahashi et al. 2007.)
  Same logic as the DDI used in Eric's HD plots.
- Latency: first time ≥2 SD from the pre-onset baseline for ≥4 consecutive bins (100 ms).
- PSTHs: 25 ms Gaussian kernel.

**JP's critique, and it is fair.** The paper is descriptive: it catalogues what 7a does but never
makes an argument. Every claim is "we measured X and here is its distribution." No perceptual
link, no decoding, no theory contact.

---

## 2. Noel & Angelaki (2022), *Annu Rev Psychol* 73:103–129 (`annurev-psych-021021-103038.pdf`)

JP's own review. The sentence that defines this project:

> "To the best of our knowledge the correlation between neural activity and heading judgments has
> not been reported in 7a."

Surrounding context that sets up the argument:

- Dorsal stream (MSTd, VIP, 7a) is broadly vestibular-responsive. 7a specifically per Avila 2019.
- The *hierarchy-of-choice-correlation* story is **not** monotonic and is contested:
  - MSTd shows weak CP (Britten & van Wezel 1998; Gu 2008); MT similar (Yu & Gu 2018).
  - VIP shows substantially **larger** CP (Chen 2013).
  - But causal work breaks the ordering: MSTd inactivation → 3× visual heading threshold
    (Gu 2012); VIP inactivation → **no effect** (Chen 2016).
  - Zaidel et al. 2017 dissociate sensory-driven from choice-driven CP: MSTd is
    heading-dominated, VIP is choice-dominated. **This is the analysis that resolves the
    paradox, and it is the one to run in 7a.**
- 7a's distinguishing anatomy: connections to retrosplenial cortex → indirect route to
  hippocampal formation (Pandya & Seltzer 1982). JP's speculation: 7a/RSC show *divergence*
  (separate linear vs angular codes) rather than convergence, because that format is what the
  hippocampal formation needs to read out.

**Implication for framing.** 7a is not "one more area in the CP ladder." It is the branch point
where egocentric parietal coding hands off to allocentric navigation coding. A finding of *weak*
choice correlation in 7a would be a positive result under that framing, not a null.

---

## 3. `MultisensoryIntegrationForSelfMotionPerception_V2.1.pdf`

Broad review; scan-level read. Relevant hooks:
- 7a mentioned only in passing (cites Avila 2019) — confirms how thin the 7a literature is.
- Contains the CP vs CI (congruency index) treatment that Gu 2008 established; §Figure 7a of that
  review shows the CP/CI relationship to replicate.

---

## 4. The three template papers (from JP's links)

| # | Ref | DOI | What to copy |
|---|-----|-----|--------------|
| A | **Gu, DeAngelis & Angelaki 2007**, *Nat Neurosci* 10:1038 — "A functional link between area MSTd and heading perception based on vestibular signals" | 10.1038/nn1935 | Psychometric vs neurometric thresholds; choice probability; CP concentrated in the most sensitive neurons; (labyrinthectomy — not available to us) |
| B | **Gu, Angelaki & DeAngelis 2008**, *Nat Neurosci* 11:1201 — "Neural correlates of multisensory cue integration in macaque MSTd" | 10.1038/nn.2191 | Congruent vs opposite cells; sub-additive combination; sensitivity *improves* in combined for congruent cells and *degrades* for opposite cells; behaviour tracks the congruent subpopulation |
| C | **Fetsch, Pouget, DeAngelis & Angelaki 2011**, *Nat Neurosci* 15:146 — "Neural correlates of reliability-based cue weighting during multisensory integration" | 10.1038/nn.2983 | Vary visual reliability (coherence); behavioural weights shift toward the reliable cue; neuronal combination weights shift the same way |

**Note on C:** this analysis *requires trials at ≥2 visual coherences*. Whether Eric's HD dataset
has that is an open question — see `03_data_questions.md`. Without it, C is not reproducible and
the paper is built on A + B.

**Note on B:** the cue-conflict (Δ) manipulation used to measure *behavioural weights* likewise
requires conflict trials (Gu 2008 used Δ = ±4°). Also unconfirmed. Without conflict trials we can
still do the *threshold* prediction (√ combination) but not the *weight* estimation from PSE
shifts.
