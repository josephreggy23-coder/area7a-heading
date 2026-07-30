# What is already in Eric's 2018 plots (`1D_HD_plots.docx`)

Extracted images live in `work/eric_plots/`. Text in the doc is sparse; most content is figures.

## Numbers stated in the doc

**1D Azi (passive):**
- Total = **209 single units**
- Tuned: Ves = 17 (8.1%), Vis = 25 (11.9%), Com = 33 (15.7%)
- Peak response: Ves 1.20 ± 0.03, Vis 1.22 ± 0.03, Com 1.21 ± 0.02

**HD (active):**
- Total = **146 single units, 256 multi-units**
- Population psychometric/neurometric panels are labelled **n = 128**
  (so ~128 units survive whatever inclusion criterion was applied — worth recovering)

## Figure-by-figure

| Image | Content |
|---|---|
| `image1.png` | **1D Azi protocol schematic.** 8 azimuths (0/45/90/…/315°) in the horizontal plane, ves/vis/com. Trial: 0.1 fixation → 0.4 static dots → **2.0 s motion** → 0.4 static dots. Example PSTH grids + polar tuning plots for 5 units. |
| `image2.png` | More 1D Azi example units (#1-1, #2-1), same format. |
| `image3.png` | 1D Azi population: (A) grand-average PSTH per modality vs baseline; (B) tuning aligned to preferred azimuth, n=209 — **combined ≥ vestibular ≈ visual**, weak modulation (4→7 spk/s); (C) distribution of preferred directions — visual over-represents 180° (contraction/backward), roughly uniform for ves/com. |
| `image4.png` | Distribution of peak response time. Vestibular is sharply bimodal (~0.6 s and ~1.4 s); visual is broad/flat. Matches the "diverse temporal dynamics" claim in bhy272. |
| `image5.png` | **DDI (discrimination index)** distributions per modality (all ≈0.4) and pairwise scatters. Com > Ves (p=0.009) and Com > Vis (p=0.010); Vis vs Ves n.s. (p=0.999). Panel C: `r_com` vs mean(ves,vis) rate — combined response is **sub-additive** (falls below the mean line at high rates). *This is already a Gu-2008-style result.* |
| `image6.png` | **The HD money figure.** (A) one session: psychometric (thresholds ves **4.05**, vis **3.50**, com **2.65**), neurometric (ves 13.65, vis 20.37, com 35.59), and heading tuning. (B) PSTHs by heading × modality. (C) population psychometric fits, n=128. (D) population neurometric fits, n=128. Plus neuronal-threshold histogram (range ~10–180°) and Com vs Vis/Ves threshold scatter. |
| `image7.png` | **Spike–choice and spike–heading correlations.** Histograms of Pearson r (mostly \|r\| < 0.15), per-neuron scatters, and **partial** correlations (`parCorr spk choice`, `parCorr spk heading`) — i.e. the Zaidel-2017 sensory/decision dissociation, already attempted. Bottom: example neuron's rate vs heading. |

## Read-outs that matter for planning

1. **Headings in HD are 0, ±1.5, ±3, ±6, ±12°** (9 values). Confirms JP's recollection.
2. **Firing rates in 7a during HD are very low — roughly 1–5 spk/s.** This is the single biggest
   constraint on the whole project. It is why single-neuron thresholds are 10–180° against a
   ~3° psychophysical threshold, i.e. **single 7a neurons are ~5–50× less sensitive than the
   monkey.** Any argument built on single-neuron sensitivity will fail. The argument has to be a
   **population** one (decoding from simultaneously recorded units, which the linear array makes
   possible).
3. **The example session's combined threshold is exactly the optimal prediction.**
   √(4.05²·3.50²/(4.05²+3.50²)) = **2.65**, and the measured combined threshold is 2.65.
   If that holds at the population level it is a clean Figure 1: *the monkeys integrate optimally.*
   (One session only — must be checked across all sessions. Suspiciously exact; verify it is not
   a fitting artefact or a value that was back-computed.)
4. **Combined neurometric threshold (35.59) is *worse* than either single cue (13.65, 20.37)** in
   that example session. That is the opposite of the behaviour and the opposite of Gu 2008's
   congruent cells. Either (a) 7a is dominated by *opposite* cells, (b) it is a noise/fitting
   artefact at these low rates, or (c) something is wrong in the ROC pipeline. **This needs to be
   the first thing checked once the data lands** — it is either a real and interesting result or a
   bug, and the two look identical from here.
5. Eric used correlation (Pearson r between rate and choice) rather than ROC/choice probability.
   The template papers all use CP. We should compute proper CP with permutation tests, and keep
   the partial-correlation version as the Zaidel-style follow-up.
