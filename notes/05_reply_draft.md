# Follow-up email to Jean-Paul

Thread: "Visuo-vestibular single neuron data in monkeys" (noelx071@umn.edu, 2026-07-29).
Joseph already replied 2026-07-30 with a one-line acknowledgement, so this follow-up carries all
the actual content.

## What this email is for

JP's original email asked for four things: read the three PDFs, explore the data structures,
implement the analyses from the three *Nature Neuroscience* papers in 7a, and deliver panels in
Illustrator plus statistics.

The follow-up has exactly three jobs. Everything else was cut.

| Job | Why it needs JP |
|---|---|
| **Get unblocked on the data** | The only hard blocker. The folder exceeds the 2 GB Dropbox Basic quota, so "Add to Dropbox" fails. |
| **Confirm the framing before building 20 panels** | Highest-value question in the email. Cheap for him to answer, expensive to get wrong. |
| **Ask about the 0°-heading trials** | He may know whether a rewarded-side / chosen-target field exists elsewhere. Not reliably answerable from the HD file alone. |

**Deliberately cut:**
- *"Is visual coherence varied?"* — `inspect_data.py` answers this in ten seconds once the file is
  in hand. Asking an advisor something you can answer yourself is a bad signal.
- *Cue-conflict trials* — no longer a blocker at all. Gu 2008 estimated neuronal weights from
  tuning curves by least squares with no conflict manipulation, so its absence costs only the
  behavioural weight comparison. See `03_analysis_plan.md`.
- *The methods-verification paragraph and the full code inventory* — accurate and mildly
  impressive, but they ask for nothing and dilute the three things that do.

**Dropbox status:** the invite was received twice (2026-07-29, from jeanpaulnc@gmail.com) and the
account is active. Do **not** ask JP to re-send it — that would look like the email wasn't checked.
The real obstacle is quota, and the email frames it that way.

*Text below mirrors the Gmail draft. Nothing has been sent.*

---

**Subject:** Re: Visuo-vestibular single neuron data in monkeys

Hi Jean-Paul,

Progress update, and one thing I'm stuck on.

The stuck part: the shared folder is bigger than my Dropbox storage quota, so it won't let me add
it. I can pull files one at a time from the web view, but if it's easy on your end, pointing me at
just the HD file — or dropping it somewhere I can download directly — would unblock me faster.
That's the only thing holding me up.

Meanwhile I've read the three papers and worked through Eric's plots, and I've built the analysis
pipeline so it's ready to run: psychometric and neurometric fitting, choice probability with
permutation tests, congruency classification, the cue-integration predictions, and a population
decoder. It runs end to end on simulated data and recovers values I know are correct, which is as
far as I can check it until I have the real thing.

Before I build out the full figure set, I want to make sure I'm aiming at the argument you
actually want. Reading Avila 2019 next to your review, the question that seems to turn this from
descriptive into an argument is: is 7a on the perceptual pathway for heading, or already on the
navigation pathway? If 7a shows choice probability concentrated in its most sensitive neurons, it
slots into the MSTd/VIP ladder. If instead it carries heading information whose trial-to-trial
fluctuations aren't read out for the choice — the Zaidel dissociation coming out opposite to VIP —
that breaks the ladder, and it fits with Avila's divergent linear/angular codes and the
retrosplenial connectivity you point to.

Eric's plots make me suspect the second. Rates in the HD task look like 1–5 spk/s and his neuronal
thresholds run 10–180° against a ~3° psychophysical threshold, so single 7a neurons are one to two
orders of magnitude less sensitive than the monkey. That pushes the argument toward population
decoding rather than single neurons — and it means the burden is showing the heading signal is
genuinely there, so a null choice probability doesn't just read as "7a is unresponsive." Does that
framing sound right to you, or would you rather I aim somewhere else? Easier to change now than
after twenty panels.

One data question I don't think I can answer myself. You were right that the choice coding is odd,
and I think the reason is that the stored value is coded relative to the stimulus rather than in
absolute space — something accuracy-like — which is why it comes out as a bell curve instead of a
sigmoid. Flipping the mapping on one side of zero recovers direction, exactly as you described.
But that means zero-heading trials can't be recovered at all: there's no correct answer at 0°, so
the information isn't in that field. That's frustrating, because 0° is where choice probability is
most informative. Do you know if there's a separate field anywhere — maybe in another file —
recording the rewarded side or which target the animal actually chose?

Last thing, a flag rather than a question. In Eric's example HD session the psychometric thresholds
are ves 4.05, vis 3.50, combined 2.65 — and the optimal prediction from the two single cues is also
exactly 2.65. That's either a very clean result or a number that got back-computed, and I'll check
it across sessions. In the same session the neurometric thresholds go the other way, combined 35.59
against 13.65 and 20.37 for the single cues, which is backwards from both the behaviour and
Gu 2008. Could be real and carried by opposite cells, could be an artefact of the low rates. First
thing I'll look at once I can run it.

Best,
Joseph
