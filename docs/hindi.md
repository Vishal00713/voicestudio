# Hindi

How to get the most natural Hindi out of VoiceStudio, and what the app does
to Hindi text on its own. Everything here works the same on macOS, Windows
and Linux; nothing needs a download beyond the engine you already use.

## Best Hindi voice clone

- **Write in Devanagari.** Roman "Hinglish" (`kya haal hai`) is read as
  English by every engine, with English vowels. Type `क्या हाल है` and the
  engine switches to Hindi pronunciation. When the language is Hindi and the
  script is mostly Roman letters, the Script box shows **Convert to native
  script**: your LLM (Settings, LLM providers; the **Script conversion**
  skill) rewrites it in Devanagari, keeping every `[tag]` and line break.
  **Undo** brings your text back, and a rewrite that loses a tag, a line or
  the script is refused rather than applied. The same button works for the
  other languages with their own script (Marathi, Bengali, Tamil, Urdu,
  Russian, …). English acronyms inside a Hindi sentence are fine: `CEO ने`
  is read as `सीईओ ने` (see the table below).
- **Pick Hindi as the language**, not Auto. Auto leaves detection to the
  engine; Hindi turns on the Hindi number, date, money and acronym reading
  below and the `से` reading of ranges (`20~30 सेकंड`).
- **Reference clip in Hindi.** Zero-shot cloning copies the delivery of the
  reference, accent included, so a 5–15 s clip of the speaker talking
  naturally *in Hindi* clones Hindi prosody; an English clip clones an
  English accent. When the sample's transcript is in another script than
  the take's language, the take comes back with a warning saying exactly
  that. Clean and close-mic is best; a little room noise is fine, because
  **Sound more human** copies it (below). No music. If you type the
  transcript yourself, type it in Devanagari.
- **Engines that clone Hindi:** the default VoiceStudio (OmniVoice) engine,
  and [VoxCPM2](engines/voxcpm2.md) (48 kHz, Hindi among its 30 languages).
  Try both with the same clip — they trade off differently per speaker.
  OmniVoice's weights are non-commercial, so with
  [Commercial-safe mode](commercial-safe-mode.md) on, VoxCPM2 is the Hindi
  engine to use. VoxCPM2 occasionally keeps talking after a line ends; such a
  take is rendered again automatically, up to twice (see its
  [run-on guard](engines/voxcpm2.md#behaviour-notes)). IndexTTS 2, CosyVoice 3, Supertonic-3 and KittenTTS do not
  do Hindi.
- **Ready-made Hindi voices.** The Voice Gallery ships **Hindi Documentary**
  (deep, dramatic nature-reel narrator) and **Hindi Comedy** (bright, quick
  "ulti duniya"-style narrator) next to the Hindi Narrator, Explainer and
  Companion. Use one as-is, or as a starting point for Voice Design.
- **Pacing and breath.** End sentences with the danda (`।`) or a full stop —
  both split chunks and set the gap length. `[pause 500ms]` works anywhere.
  Turn on **Sound more human** in the Audio quality popover: pauses get the
  reference clip's own room sound instead of digital silence, and the
  speaker's own inhales from the clip are placed before sentences
  ([audio-quality.md](audio-quality.md)).
- **Pronunciation.** For a loanword or a name the engine gets wrong, add a
  respelling to the pronunciation dictionary or inline as `[[...]]`
  ([expressive-speech.md](expressive-speech.md)).

## What Hindi text normalization does

A Hindi script that reaches the engine with digits in it gets those digits
read in English, or garbled. With the language set to Hindi (display name
or `hi`, `hi-IN`), VoiceStudio rewrites them the way a Hindi reader says
them, before synthesis and before the pronunciation dictionary:

| You write | The engine hears |
|---|---|
| `42 लोग` | `बयालीस लोग` |
| `150 रुपये` | `एक सौ पचास रुपये` |
| `1984 की कहानी` | `उन्नीस सौ चौरासी की कहानी` (1100–1999 use the spoken सौ form) |
| `2026 में` | `दो हज़ार छब्बीस में` |
| `150000` / `1,50,000` | `एक लाख पचास हज़ार` |
| `5,00,00,000` | `पाँच करोड़` (Indian grouping up to 99 crore) |
| `1.5 घंटे`, `2.5 करोड़`, `3.5 किलोमीटर` | `डेढ़ घंटे`, `ढाई करोड़`, `साढ़े तीन किलोमीटर` |
| `0.5 किलो`, `2.25 लाख`, `1.75 करोड़` | `आधा किलो`, `सवा दो लाख`, `पौने दो करोड़` |
| `3.7 किलोमीटर` | `तीन दशमलव सात किलोमीटर` (no spoken fraction word exists) |
| `₹500` / `Rs. 500` / `रु. 500` | `पाँच सौ रुपये` (`₹1` → `एक रुपया`, `₹5.50` → `पाँच रुपये पचास पैसे`) |
| `₹2.5 लाख`, `₹5 करोड़` | `ढाई लाख रुपये`, `पाँच करोड़ रुपये` |
| `$5`, `$2 मिलियन` | `पाँच डॉलर`, `दो मिलियन डॉलर` |
| `15/08/2026` (also `-` or `.`) | `पंद्रह अगस्त दो हज़ार छब्बीस` |
| `98765 43210`, `+91 98765 43210` | `नौ आठ सात छह पाँच, चार तीन दो एक शून्य` (digit by digit, two groups) |
| `10:30 बजे` | `साढ़े दस बजे` (`1:30` → `डेढ़`, `2:30` → `ढाई`, `10:15` → `सवा दस`, `10:45` → `पौने ग्यारह`) |
| `10:00`, `05:30` | `दस बजे`, `साढ़े पाँच` |
| `10:20 बजे` | `दस बजकर बीस मिनट पर` |
| `5वीं कक्षा` / `1वां दिन` | `पाँचवीं कक्षा` / `पहला दिन` |
| `50%`, `2.5%` | `पचास प्रतिशत`, `ढाई प्रतिशत` |
| `२०२६` | `दो हज़ार छब्बीस` (Devanagari digits are digits) |
| `20~30 सेकंड` | `बीस से तीस सेकंड` |
| `CEO ने`, `UPI से`, `ISRO का` | `सीईओ ने`, `यूपीआई से`, `इसरो का` |
| `डॉ. शर्मा`, `Dr. शर्मा`, `प्रो.`, `Mr.`, `Mrs.`, `Ms.` | `डॉक्टर शर्मा`, `प्रोफ़ेसर`, `मिस्टर`, `मिसेज़`, `मिस` |

The same conservative rules as every other language apply: `1,000` (plain
thousands grouping), `3-5`, `007`, ungrouped numbers of seven or more digits
that are not a mobile number, digits glued to a word (`कक्षा10`),
`12:34:56`, impossible dates (`31/02/2026`), two-digit years and anything
inside `[...]` are left exactly as written. Acronyms are only respelled
inside a Hindi sentence, so an English line in capitals keeps its English
reading. The pass is idempotent and can be switched off with
`OMNIVOICE_TEXT_NORMALIZATION=0`.

## Measuring a change

`scripts/eval_hindi.py` renders a fixed Hindi set
(`scripts/eval/hindi_eval_set.json`: numbers, money and dates, acronyms,
questions, long sentences, narration, names) through a running backend and
transcribes every take back with the installed faster-whisper model. It
reports the character error rate per category, the longest run of digital
silence and the pause floor, and keeps the WAVs for listening. It removes
its takes from the app's history afterwards. Run it with the app's own
Python runtime while the backend is up:

```sh
python scripts/eval_hindi.py --output eval-runs/doc --engine voxcpm2 \
    --profile-id <voice id> --humanize
```

`--client-normalize` normalizes the text with your checkout before sending
it, to score a normalization change against a backend that still runs older
code. Speaker similarity and MOS predictors need extra model downloads and
are not part of it; compare the WAVs by ear as well.
