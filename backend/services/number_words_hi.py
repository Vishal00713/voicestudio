"""Hindi (hi) text verbalization — a native backend for text_normalization.

``num2words`` has no Hindi locale (it ships bn/kn/te but not hi, ml or ta),
so a Hindi script reached the engine with raw digits: OmniVoice reads "2026"
in English, or garbles it, in the middle of a Devanagari sentence — the
single most audible "this is a machine" moment in a Hindi take. This module
renders what a Hindi news reader actually says:

  * cardinals 0–99,99,99,999 with the irregular 1–99 table (इक्कीस, बाईस, …),
    सौ, हज़ार, लाख (10^5) and करोड़ (10^7): 150000 → एक लाख पचास हज़ार,
    5,00,00,000 → पाँच करोड़;
  * the spoken "सौ" form for 1100–1999 — years and amounts alike — because
    1984 is उन्नीस सौ चौरासी and 1500 रुपये is पंद्रह सौ रुपये, never
    "एक हज़ार नौ सौ चौरासी"; 2000 and up stay on हज़ार (दो हज़ार छब्बीस);
  * the fraction words a speaker uses before a unit: 1.5 घंटे → डेढ़ घंटे,
    2.5 करोड़ → ढाई करोड़, 3.5 किलोमीटर → साढ़े तीन किलोमीटर, 2.25 लाख →
    सवा दो लाख, 1.75 करोड़ → पौने दो करोड़, 0.5 किलो → आधा किलो (also for
    percentages: 2.5% → ढाई प्रतिशत); any other decimal is read digit by
    digit after दशमलव;
  * money: ₹500, Rs. 500, रु. 500 → … रुपये (₹1 → एक रुपया, ₹5.50 → … पैसे),
    ₹2.5 लाख → ढाई लाख रुपये, $5 → पाँच डॉलर, $2 मिलियन → दो मिलियन डॉलर;
  * dates written DD/MM/YYYY (also with - or .): 15/08/2026 → पंद्रह अगस्त
    दो हज़ार छब्बीस; impossible dates are left alone;
  * Indian mobile numbers digit by digit in two groups of five, the way they
    are read out: 98765 43210, +91 98765 43210, 09876543210;
  * clock times the way they are spoken: 10:30 → साढ़े दस, 1:30 → डेढ़,
    2:30 → ढाई, 10:15 → सवा दस, 10:45 → पौने ग्यारह, 10:00 → दस बजे,
    10:20 → दस बजकर बीस मिनट (a following "बजे" is folded in, becoming
    "पर" after the बजकर form); 05:30 works like 5:30;
  * ordinals written with a digit: 5वां → पाँचवां, 1वीं → पहली, 4वें → चौथे;
  * Indian digit grouping (1,50,000) which the generic conservative rules
    leave alone as "ambiguous" but is unambiguous in Hindi text;
  * Latin acronyms inside a Hindi sentence in Devanagari letter names, so the
    voice does not switch accent mid-sentence: CEO ने → सीईओ ने, UPI से →
    यूपीआई से; the ones said as a word keep that word (ISRO → इसरो).

Scope matches the conservative rules of :mod:`services.text_normalization`:
bracket spans untouched, ungrouped numbers of seven or more digits are IDs
(left alone unless they are a mobile number), any failure leaves the
occurrence as written. Pure Python, no dependencies; every rendering contains
no digits or Latin capitals, so a second pass is a no-op.
"""
from __future__ import annotations

import datetime
import re
import unicodedata

# ── Tables ───────────────────────────────────────────────────────────────────

_BELOW_HUNDRED = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]

HUNDRED = "सौ"
THOUSAND = "हज़ार"
LAKH = "लाख"
CRORE = "करोड़"
DECIMAL_WORD = "दशमलव"
PERCENT_WORD = "प्रतिशत"
RUPEE = "रुपये"
RUPEE_ONE = "रुपया"
PAISE = "पैसे"
DOLLAR = "डॉलर"
CENTS = "सेंट"
PLUS = "प्लस"
OCLOCK = "बजे"
PAST = "बजकर"
MINUTES = "मिनट"

MAX_CARDINAL = 999_999_999  # 99,99,99,999 — ninety-nine crore and change

_MONTHS = [
    "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
]

# The idiomatic quarter forms: (quarter past, half past, quarter to) are
# सवा / साढ़े / पौने + hour, except the hours that have their own word.
_HALF_PAST_SPECIAL = {1: "डेढ़", 2: "ढाई"}

# Ordinal stems for the irregular ordinals; everything else is cardinal + suffix.
_ORDINAL_STEMS = {1: "पहल", 2: "दूसर", 3: "तीसर", 4: "चौथ", 6: "छठ"}
_ORDINAL_ENDINGS = {"वां": "ा", "वाँ": "ा", "वीं": "ी", "वें": "े"}

# English letter names as Hindi writes them (सीईओ, यूपीआई, जीएसटी).
_LETTERS = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ़", "G": "जी",
    "H": "एच", "I": "आई", "J": "जे", "K": "के", "L": "एल", "M": "एम", "N": "एन",
    "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर", "S": "एस", "T": "टी", "U": "यू",
    "V": "वी", "W": "डब्ल्यू", "X": "एक्स", "Y": "वाई", "Z": "ज़ेड",
}

# Acronyms Hindi speakers say as a word, not letter by letter.
_ACRONYM_WORDS = {
    "AIIMS": "एम्स", "ASEAN": "आसियान", "COVID": "कोविड", "FIFA": "फ़ीफ़ा",
    "ISRO": "इसरो", "LASER": "लेज़र", "NASA": "नासा", "NATO": "नाटो",
    "NEET": "नीट", "NITI": "नीति", "OPEC": "ओपेक", "PIN": "पिन", "RADAR": "रडार",
    "RAM": "रैम", "SAARC": "सार्क", "SEBI": "सेबी", "SIM": "सिम",
    "UNESCO": "यूनेस्को", "UNICEF": "यूनिसेफ़",
}


# ── Cardinals ────────────────────────────────────────────────────────────────

def _below_thousand(n: int) -> str:
    """0-999: hundreds as "<unit> सौ", then the irregular 1-99 word."""
    hundreds, rest = divmod(n, 100)
    parts: list[str] = []
    if hundreds:
        parts.append(f"{_BELOW_HUNDRED[hundreds]} {HUNDRED}")
    if rest or not parts:
        parts.append(_BELOW_HUNDRED[rest])
    return " ".join(parts)


def _indian_groups(n: int) -> str:
    """Any n ≥ 1000 as करोड़ / लाख / हज़ार / the rest, without the सौ form."""
    crores, rest = divmod(n, 10_000_000)
    lakhs, rest = divmod(rest, 100_000)
    thousands, units = divmod(rest, 1000)
    parts: list[str] = []
    for count, word in ((crores, CRORE), (lakhs, LAKH), (thousands, THOUSAND)):
        if count:
            parts.append(f"{_BELOW_HUNDRED[count]} {word}")
    if units:
        parts.append(_below_thousand(units))
    return " ".join(parts)


def cardinal(n: int) -> str:
    """Render an integer 0 ≤ n ≤ 99,99,99,999 as Hindi words."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("cardinal() expects an int")
    if n < 0 or n > MAX_CARDINAL:
        raise ValueError("cardinal() supports 0..999999999")
    if n < 1000:
        return _below_thousand(n)
    if 1100 <= n <= 1999:
        # Spoken form for years and amounts: उन्नीस सौ चौरासी, पंद्रह सौ.
        hundreds, rest = divmod(n, 100)
        head = f"{_BELOW_HUNDRED[hundreds]} {HUNDRED}"
        return f"{head} {_BELOW_HUNDRED[rest]}" if rest else head
    return _indian_groups(n)


def decimal(int_part: int, frac_digits: str) -> str:
    """3.14 → तीन दशमलव एक चार (fraction digits are read one by one)."""
    digits = " ".join(_BELOW_HUNDRED[int(d)] for d in frac_digits)
    return f"{cardinal(int_part)} {DECIMAL_WORD} {digits}"


def spoken_fraction(int_part: int, frac_digits: str) -> str | None:
    """The word a speaker uses for a half or a quarter, or ``None``.

    .5 → आधा / डेढ़ / ढाई / साढ़े N; .25 → सवा / सवा N; .75 → पौने N+1.
    Anything else has no idiomatic form and is read with दशमलव instead.
    """
    frac = frac_digits.rstrip("0")
    n = int_part
    if frac == "5":
        if n == 0:
            return "आधा"
        if n in _HALF_PAST_SPECIAL:
            return _HALF_PAST_SPECIAL[n]
        return f"साढ़े {cardinal(n)}"
    if frac == "25" and n >= 1:
        return "सवा" if n == 1 else f"सवा {cardinal(n)}"
    if frac == "75" and n >= 1:
        return f"पौने {cardinal(n + 1)}"
    return None


def _amount_words(raw: str) -> str:
    """'5' → पाँच, '2.5' → ढाई, '1.2' → एक दशमलव दो, '1,50,000' → एक लाख पचास हज़ार."""
    raw = raw.replace(",", "")
    if "." not in raw:
        return cardinal(int(raw))
    a, b = raw.split(".", 1)
    return spoken_fraction(int(a), b) or decimal(int(a), b)


def percent(raw: str) -> str:
    """'50' → पचास प्रतिशत; '2.5' → ढाई प्रतिशत; '3.7' → तीन दशमलव सात प्रतिशत."""
    return f"{_amount_words(raw)} {PERCENT_WORD}"


# ── Hindi-specific forms ─────────────────────────────────────────────────────

def _grouped_int(raw: str) -> int:
    """'1,50,000' → 150000. Raises for anything cardinal() cannot say."""
    return int(raw.replace(",", ""))


def rupees(raw_amount: str, paise: str | None = None) -> str:
    """'500' → पाँच सौ रुपये; '1' → एक रुपया; ('5', '50') → पाँच रुपये पचास पैसे."""
    amount = _grouped_int(raw_amount)
    paise_n = int(paise) if paise else 0
    if amount == 1 and not paise_n:
        return f"{cardinal(1)} {RUPEE_ONE}"
    words = f"{cardinal(amount)} {RUPEE}" if amount or not paise_n else ""
    if paise_n:
        tail = f"{cardinal(paise_n)} {PAISE}"
        return f"{words} {tail}" if words else tail
    return words


def dollars(raw_amount: str, cents: str | None = None) -> str:
    """'5' → पाँच डॉलर; ('5', '50') → पाँच डॉलर पचास सेंट."""
    amount = _grouped_int(raw_amount)
    cents_n = int(cents) if cents else 0
    words = f"{cardinal(amount)} {DOLLAR}" if amount or not cents_n else ""
    if cents_n:
        tail = f"{cardinal(cents_n)} {CENTS}"
        return f"{words} {tail}" if words else tail
    return words


def date(day: int, month: int, year: int) -> str:
    """15, 8, 2026 → पंद्रह अगस्त दो हज़ार छब्बीस. Raises for impossible dates."""
    if not 1900 <= year <= 2099:
        raise ValueError("year outside 1900..2099")
    datetime.date(year, month, day)  # validates day/month, raises ValueError
    return f"{cardinal(day)} {_MONTHS[month - 1]} {cardinal(year)}"


def digits(raw: str) -> str:
    """'98765' → नौ आठ सात छह पाँच."""
    return " ".join(_BELOW_HUNDRED[int(d)] for d in raw)


def clock(hour: int, minute: int, *, spoken_oclock: bool = False) -> str:
    """The spoken time for ``hour``:``minute`` (24-hour input).

    ``spoken_oclock`` says a "बजे" followed the digits in the text; it is
    folded into the forms that carry their own (the quarter forms keep it,
    "बजकर … मिनट" replaces it, "N बजे" is the on-the-hour form either way).
    """
    h = hour % 12 or 12
    nxt = h % 12 + 1
    suffix = f" {OCLOCK}" if spoken_oclock else ""
    if minute == 0:
        return f"{_BELOW_HUNDRED[h]} {OCLOCK}"
    if minute == 15:
        return f"सवा {_BELOW_HUNDRED[h]}{suffix}"
    if minute == 30:
        return f"{_HALF_PAST_SPECIAL.get(h, 'साढ़े ' + _BELOW_HUNDRED[h])}{suffix}"
    if minute == 45:
        return f"पौने {_BELOW_HUNDRED[nxt]}{suffix}"
    # "10:20 बजे मिलेंगे" → "दस बजकर बीस मिनट पर मिलेंगे": the postposition
    # that "बजे" carried has to survive the rewrite.
    tail = " पर" if spoken_oclock else ""
    return f"{_BELOW_HUNDRED[h]} {PAST} {_BELOW_HUNDRED[minute]} {MINUTES}{tail}"


def ordinal(n: int, suffix: str) -> str:
    """5 + वां → पाँचवां; the irregular 1/2/3/4/6 take their own stems."""
    ending = _ORDINAL_ENDINGS[suffix]
    stem = _ORDINAL_STEMS.get(n)
    if stem is not None:
        return stem + ending
    return cardinal(n) + suffix


def acronym(letters: str) -> str:
    """'CEO' → सीईओ; 'ISRO' → इसरो."""
    return _ACRONYM_WORDS.get(letters) or "".join(_LETTERS[ch] for ch in letters)


# ── Patterns ─────────────────────────────────────────────────────────────────
#
# text_normalization runs these before its generic percent/decimal/integer
# passes, with the same boundary guards as its own regexes. Each entry is
# ``(regex, render)`` or ``(regex, render, zero_guard)``: with the guard on
# (the default) a group 1 that starts with a zero is a code ("007") and is
# left alone; dates, times and phone numbers turn it off because "05/08",
# "05:30" and "098765…" are ordinary ways to write them. Order matters:
# money, dates, phones and times claim their digits first.

# Every spelling a Devanagari word can arrive in: the request path applies
# NFC, which *decomposes* the nukta letters (U+0958–U+095F are composition
# exclusions), while text typed elsewhere may carry the precomposed ones.
_NUKTA_PRECOMPOSED = {
    "\u0915\u093c": "\u0958", "\u0916\u093c": "\u0959", "\u0917\u093c": "\u095a",
    "\u091c\u093c": "\u095b", "\u0921\u093c": "\u095c", "\u0922\u093c": "\u095d",
    "\u092b\u093c": "\u095e", "\u092f\u093c": "\u095f",
}


def _spellings(word: str) -> set[str]:
    decomposed = unicodedata.normalize("NFC", word)
    composed = decomposed
    for seq, ch in _NUKTA_PRECOMPOSED.items():
        composed = composed.replace(seq, ch)
    return {decomposed, composed}


def _alternation(words) -> str:
    forms = sorted({s for w in words for s in _spellings(w)}, key=len, reverse=True)
    return "|".join(re.escape(f) for f in forms)


#: A word boundary that also refuses Devanagari: Python's ``\w`` excludes the
#: vowel signs, so "करोड़ों" would otherwise end a match after "करोड़".
_DEV_END = r"(?![\w\u0900-\u097F])"

_SCALES = ("लाख", "करोड़", "हज़ार", "हजार", "अरब", "मिलियन", "बिलियन")
_UNITS = _SCALES + (
    "घंटे", "घंटा", "घंटों", "साल", "वर्ष", "महीने", "महीना", "दिन", "मिनट",
    "सेकंड", "किलो", "किलोग्राम", "ग्राम", "किलोमीटर", "मीटर", "सेंटीमीटर",
    "लीटर", "फ़ीट", "फीट", "इंच", "टन", "गुना", "प्रतिशत", "फ़ीसदी", "फीसदी",
    "रुपये", "डॉलर", "स्टार", "ओवर",
)

_CURRENCY = r"(\u20b9|[Rr]s\.?|रु\.?|\$)"
_AMOUNT = r"(\d{1,2}(?:,\d{2})+,\d{3}|\d{1,6})"

_MONEY_SCALE_RE = re.compile(
    r"(?<![\w.,:/$%+-])" + _CURRENCY + r"\s?(\d{1,6}(?:\.\d{1,2})?)\s*("
    + _alternation(_SCALES) + r")" + _DEV_END
)
_MONEY_RE = re.compile(
    r"(?<![\w.,:/$%+-])" + _CURRENCY + r"\s?" + _AMOUNT
    + r"(?:\.(\d{2}))?(?![\w:/%+-])(?![.,]\d)"
)
_DATE_RE = re.compile(
    r"(?<![\w.,:/$%+-])(\d{1,2})([/.\-])(\d{1,2})\2(\d{4})(?![\w:/%+-])(?![.,]\d)"
)
_PHONE_RE = re.compile(
    r"(?<![\w.,:/$%+-])(\+91[\s-]?|0)?([6-9]\d{4})[\s-]?(\d{5})(?![\w:/%+-])(?![.,]\d)"
)
_TIME_RE = re.compile(
    r"(?<![\w:.,/$%+-])([01]?\d|2[0-3]):([0-5]\d)(?![\w:/%+-])(?![.,]\d)(\s*बजे(?!\w))?"
)
_FRACTION_UNIT_RE = re.compile(
    r"(?<![\w.,:/$%+-])(\d{1,6})\.(5|50|25|75)(?=\s*(?:" + _alternation(_UNITS) + r")"
    + _DEV_END + r")"
)
_ORDINAL_RE = re.compile(r"(?<![\w.,:/$%+-])(\d{1,4})(वां|वाँ|वीं|वें)(?![\w])")
_GROUPED_RE = re.compile(
    r"(?<![\w.,:/$%-])(\d{1,2}(?:,\d{2})+,\d{3})(?![\w:/%-])(?![.,]\d)"
)
_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9.&])([A-Z]{2,6})(?![A-Za-z0-9&])")


def _is_rupee(sign: str) -> bool:
    return sign != "$"


def _money_scale(m: re.Match) -> str:
    sign, raw, scale = m.group(1), m.group(2), m.group(3)
    unit = RUPEE if _is_rupee(sign) else DOLLAR
    return f"{_amount_words(raw)} {scale} {unit}"


def _money(m: re.Match) -> str:
    if _is_rupee(m.group(1)):
        return rupees(m.group(2), m.group(3))
    return dollars(m.group(2), m.group(3))


def _date(m: re.Match) -> str:
    return date(int(m.group(1)), int(m.group(3)), int(m.group(4)))


def _phone(m: re.Match) -> str:
    prefix = (m.group(1) or "").strip(" -")
    head = {"": "", "0": f"{_BELOW_HUNDRED[0]} ", "+91": f"{PLUS} {digits('91')}, "}[prefix]
    return f"{head}{digits(m.group(2))}, {digits(m.group(3))}"


def _time(m: re.Match) -> str:
    return clock(int(m.group(1)), int(m.group(2)), spoken_oclock=bool(m.group(3)))


def _fraction_unit(m: re.Match) -> str:
    words = spoken_fraction(int(m.group(1)), m.group(2))
    if words is None:  # unreachable for the captured fractions; be explicit
        raise ValueError("no spoken fraction")
    return words


def _ordinal(m: re.Match) -> str:
    return ordinal(int(m.group(1)), m.group(2))


def _grouped(m: re.Match) -> str:
    return cardinal(_grouped_int(m.group(1)))


def _neighbour_is_devanagari(text: str, start: int, end: int) -> bool:
    """True when the nearest letter before or after a span is Devanagari."""
    for i in range(start - 1, -1, -1):
        ch = text[i]
        if ch.isalpha() or unicodedata.category(ch).startswith("M"):
            return "\u0900" <= ch <= "\u097f"
    for i in range(end, len(text)):
        ch = text[i]
        if ch.isalpha() or unicodedata.category(ch).startswith("M"):
            return "\u0900" <= ch <= "\u097f"
    return False


def _acronym(m: re.Match) -> str:
    # Only inside Hindi: an all-caps word in an English sentence ("I LOVE
    # THIS") is a word, not an acronym, and must keep its English reading.
    if not _neighbour_is_devanagari(m.string, m.start(), m.end()):
        return m.group(0)
    return acronym(m.group(1))


PATTERNS: list[tuple] = [
    (_MONEY_SCALE_RE, _money_scale, False),
    (_MONEY_RE, _money, False),
    (_DATE_RE, _date, False),
    (_PHONE_RE, _phone, False),
    (_TIME_RE, _time, False),
    (_FRACTION_UNIT_RE, _fraction_unit),
    (_ORDINAL_RE, _ordinal),
    (_GROUPED_RE, _grouped),
    (_ACRONYM_RE, _acronym),
]

__all__ = [
    "PATTERNS", "acronym", "cardinal", "clock", "date", "decimal", "digits",
    "dollars", "ordinal", "percent", "rupees", "spoken_fraction",
]

