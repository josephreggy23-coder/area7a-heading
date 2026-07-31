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

Status: read the three papers, worked through Eric's plots, and the analysis pipeline is built and
tested on simulated data. Ready to run.

Blocker: the Dropbox folder is bigger than my storage quota, so it won't add. I can pull files one
at a time from the web, but if it's easy, could you point me at just the HD file?

Framing check before I build the figures. Reading Avila next to your review, the question that
makes this an argument seems to be whether 7a is on the perceptual pathway or already on the
navigation pathway. Eric's plots suggest the second. Rates are 1-5 spk/s and his neuronal
thresholds run 10-180 deg against ~3 deg behavior, so single neurons are far less sensitive than
the monkey. That pushes toward population decoding, and it means we'd have to show heading is
decodable, so a null CP doesn't just read as "7a is unresponsive." Is that the argument you want,
or should I aim elsewhere?

Data question. The choice field looks coded relative to the stimulus rather than in absolute
space, which is why it plots as a bell curve. Flipping the sign on one side recovers direction, as
you said. But 0 deg then has no recoverable answer, and 0 deg is where CP is most informative. Is
there a field anywhere with the rewarded side or the chosen target?

Two things I'll check once I can run it. Eric's example session has psychometric thresholds 4.05,
3.50, 2.65, where the optimal prediction is also exactly 2.65. And his neurometric thresholds go
the wrong way: combined 35.59 against 13.65 and 20.37 for the single cues.

Joseph
