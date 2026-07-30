# Draft reply to Jean-Paul

*Not sent. Review and edit before sending — particularly the tone of the section on Eric's
combined-neurometric result, and decide whether to include the repo link.*

---

Hi Jean-Paul,

Thanks — this is a great project and I'm glad to be picking it up. I've read the three PDFs and
gone through Eric's 2018 plots, and I've started building the analysis pipeline. A few things:

**Dropbox access still isn't working.** I don't see an invitation I can act on, so I haven't been
able to get at the data yet. Could you re-send it to larpsystem@gmail.com? In the meantime I've
been working from the papers and from `1D_HD_plots.docx`, which turned out to be quite
informative.

**On the framing.** I take your point about Avila 2019 being descriptive. Reading it alongside
your review, the question that seems to make the 7a data argue rather than describe is: *is 7a on
the perceptual pathway for heading, or already on the navigation pathway?* Set up that way, both
outcomes are informative. If 7a shows significant choice probability concentrated in its most
sensitive neurons, with congruent cells improving under cue combination, it slots into the
MSTd→VIP ladder. If instead it carries heading information whose trial-to-trial fluctuations
aren't read out for the choice — the Zaidel 2017 dissociation coming out opposite to VIP — that's
the more interesting result, because combined with Avila's *divergent* linear/angular codes and
the retrosplenial connectivity it says 7a's heading signal is formatted for allocentric readout
rather than for a left/right discrimination.

Eric's plots make me suspect the second. Which means the burden would be to show the heading
signal is genuinely there, so a null CP doesn't just read as "7a is unresponsive." I think a
population decoder that recovers heading well while CP stays at chance is the strongest version of
that argument — and the 16-channel U-Probe recordings should support it.

**On the choice coding.** Your recollection looks right, and I think there's a clean explanation.
A bell shape rather than a sigmoid is the signature of a variable coded *relative to the stimulus*
rather than in absolute space — i.e. something accuracy-like. If the stored value means "correct,"
then P(that value) against heading is U-shaped; if it means "error," it's the inverted-U you
remember. Either way absolute direction comes back by flipping the mapping on one side of zero,
exactly as you described. I've written a resolver that fits all four candidate mappings and picks
by likelihood, reporting monotonicity and symmetry diagnostics so we can see *why* it chose rather
than taking it on faith. It's tested against simulated data encoded the awkward way, and recovers
the truth.

One consequence worth flagging early: **if the coding really is stimulus-relative, zero-heading
trials can't be decoded at all** — there's no correct answer at 0°, so the information is simply
gone. That's awkward, because 0° is where choice probability is most informative (no stimulus
drive to remove). If there's a separate field somewhere recording the rewarded side or the actual
target the animal chose on those trials, it would rescue the most valuable trials in the dataset.
Do you know if something like that exists?

**Two questions that gate whole analyses:**

1. **Are there cue-conflict (Δ) trials?** Gu 2008 used Δ = ±4° to measure the *weight* the animal
   placed on each cue. Without conflict trials we can still test the threshold prediction, but not
   the weight prediction — which is half of that paper's argument.
2. **Is visual coherence varied?** The Fetsch 2011 result *is* the reliability manipulation. If
   every trial is at one coherence, that paper isn't reproducible here and the story rests on the
   other two. Better to know now than after building toward it.

**Two things in Eric's plots I want to flag.** In his example HD session the psychometric
thresholds are ves 4.05°, vis 3.50°, combined 2.65° — and the optimal prediction from the two
single cues is 2.65°. That's exact to three digits, which is either a lovely result or a number
that got back-computed somewhere; worth checking across all sessions. Separately, the *neurometric*
thresholds in that session go the other way: ves 13.65, vis 20.37, combined **35.59**. Combined
being worse than either single cue is the opposite of the behaviour and the opposite of Gu 2008's
congruent cells. That's either a real effect carried by opposite cells, or a bug in the ROC
pipeline — and the two look identical from where I'm sitting. It's the first thing I'll check once
I have the data.

**Where things stand.** I've set up an analysis repo with the pieces the three papers need:
psychometric fitting, the choice-coding resolver, ROC and antineuron neurometric functions, choice
probability with permutation tests, DDI and congruency classification, partial correlations, and
the cue-integration predictions. It runs end-to-end on simulated data and recovers known ground
truth, and it exports panels as vector PDF/SVG with live text so they open cleanly in Illustrator,
with a CSV of the plotted values and a JSON of the stats beside each one. When the data arrives
the only thing that changes is the loader.

Happy to walk through any of this whenever suits.

Best,
Joseph
