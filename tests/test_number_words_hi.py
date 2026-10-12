"""Hindi text verbalizer (``services/number_words_hi.py``).

Pinned renderings are what a Hindi news reader says, not what a lookup table
would mechanically produce — the सौ form for 1100–1999, डेढ़/ढाई/साढ़े/सवा/पौने
before units and on the clock, पहला/दूसरी ordinals, एक रुपया vs रुपये, and
acronyms in the letter names Hindi writes.
"""
import random
import unicodedata

import pytest

from services import number_words_hi as hi
from services.text_normalization import normalize_text


@pytest.mark.parametrize("n, expected", [
    (0, "शून्य"),
    (1, "एक"),
    (5, "पाँच"),
    (19, "उन्नीस"),
    (21, "इक्कीस"),
    (42, "बयालीस"),
    (99, "निन्यानवे"),
    (100, "एक सौ"),
    (101, "एक सौ एक"),
    (150, "एक सौ पचास"),
    (999, "नौ सौ निन्यानवे"),
    (1000, "एक हज़ार"),
    (1001, "एक हज़ार एक"),
    (1099, "एक हज़ार निन्यानवे"),
    (1100, "ग्यारह सौ"),
    (1200, "बारह सौ"),
    (1500, "पंद्रह सौ"),
    (1984, "उन्नीस सौ चौरासी"),
    (1999, "उन्नीस सौ निन्यानवे"),
    (2000, "दो हज़ार"),
    (2026, "दो हज़ार छब्बीस"),
    (2200, "दो हज़ार दो सौ"),
    (10000, "दस हज़ार"),
    (25000, "पच्चीस हज़ार"),
    (99999, "निन्यानवे हज़ार नौ सौ निन्यानवे"),
    (100000, "एक लाख"),
    (150000, "एक लाख पचास हज़ार"),
    (250500, "दो लाख पचास हज़ार पाँच सौ"),
    (999999, "नौ लाख निन्यानवे हज़ार नौ सौ निन्यानवे"),
    (1250000, "बारह लाख पचास हज़ार"),
    (10000000, "एक करोड़"),
    (10001500, "एक करोड़ एक हज़ार पाँच सौ"),
    (50000000, "पाँच करोड़"),
    (123456789, "बारह करोड़ चौंतीस लाख छप्पन हज़ार सात सौ नवासी"),
    (999999999, "निन्यानवे करोड़ निन्यानवे लाख निन्यानवे हज़ार नौ सौ निन्यानवे"),
])
def test_cardinal(n, expected):
    assert unicodedata.normalize("NFC", hi.cardinal(n)) == unicodedata.normalize("NFC", expected)


def test_cardinal_covers_the_whole_range_without_digits():
    for n in range(1_000_000):
        word = hi.cardinal(n)
        assert word and not any(ch.isdigit() for ch in word), n
    rng = random.Random(7)
    for n in [rng.randrange(1_000_000, hi.MAX_CARDINAL + 1) for _ in range(20000)] + [
        hi.MAX_CARDINAL, 10_000_000, 10_000_001, 99_999_999,
    ]:
        word = hi.cardinal(n)
        assert word and not any(ch.isdigit() for ch in word), n


def test_cardinal_rejects_out_of_range_and_non_ints():
    with pytest.raises(ValueError):
        hi.cardinal(hi.MAX_CARDINAL + 1)
    with pytest.raises(ValueError):
        hi.cardinal(-1)
    with pytest.raises(TypeError):
        hi.cardinal(True)


@pytest.mark.parametrize("n, frac, expected", [
    (0, "5", "आधा"),
    (1, "5", "डेढ़"),
    (2, "5", "ढाई"),
    (3, "5", "साढ़े तीन"),
    (3, "50", "साढ़े तीन"),
    (1, "25", "सवा"),
    (2, "25", "सवा दो"),
    (1, "75", "पौने दो"),
    (9, "75", "पौने दस"),
    (0, "25", None),
    (0, "75", None),
    (3, "7", None),
    (3, "14", None),
])
def test_spoken_fraction(n, frac, expected):
    assert hi.spoken_fraction(n, frac) == expected


def test_decimal_and_percent():
    assert hi.decimal(3, "5") == "तीन दशमलव पाँच"
    assert hi.decimal(3, "14") == "तीन दशमलव एक चार"
    assert hi.percent("50") == "पचास प्रतिशत"
    assert hi.percent("2.5") == "ढाई प्रतिशत"
    assert hi.percent("0.5") == "आधा प्रतिशत"
    assert hi.percent("3.7") == "तीन दशमलव सात प्रतिशत"


@pytest.mark.parametrize("amount, paise, expected", [
    ("1", None, "एक रुपया"),
    ("500", None, "पाँच सौ रुपये"),
    ("1,50,000", None, "एक लाख पचास हज़ार रुपये"),
    ("5", "50", "पाँच रुपये पचास पैसे"),
    ("1", "25", "एक रुपये पच्चीस पैसे"),
    ("0", "75", "पचहत्तर पैसे"),
])
def test_rupees(amount, paise, expected):
    assert hi.rupees(amount, paise) == expected


@pytest.mark.parametrize("hour, minute, oclock, expected", [
    (10, 0, False, "दस बजे"),
    (10, 0, True, "दस बजे"),
    (10, 15, False, "सवा दस"),
    (10, 15, True, "सवा दस बजे"),
    (10, 30, True, "साढ़े दस बजे"),
    (1, 30, True, "डेढ़ बजे"),
    (2, 30, False, "ढाई"),
    (13, 30, False, "डेढ़"),
    (10, 45, True, "पौने ग्यारह बजे"),
    (12, 45, False, "पौने एक"),
    (0, 30, False, "साढ़े बारह"),
    (10, 20, False, "दस बजकर बीस मिनट"),
    (10, 20, True, "दस बजकर बीस मिनट पर"),
    (23, 5, False, "ग्यारह बजकर पाँच मिनट"),
])
def test_clock(hour, minute, oclock, expected):
    assert hi.clock(hour, minute, spoken_oclock=oclock) == expected


@pytest.mark.parametrize("n, suffix, expected", [
    (1, "वां", "पहला"),
    (1, "वाँ", "पहला"),
    (1, "वीं", "पहली"),
    (2, "वीं", "दूसरी"),
    (3, "वें", "तीसरे"),
    (4, "वां", "चौथा"),
    (6, "वां", "छठा"),
    (5, "वां", "पाँचवां"),
    (5, "वीं", "पाँचवीं"),
    (25, "वें", "पच्चीसवें"),
    (100, "वां", "एक सौवां"),
])
def test_ordinal(n, suffix, expected):
    assert hi.ordinal(n, suffix) == expected


def test_date_validates():
    assert hi.date(15, 8, 2026) == "पंद्रह अगस्त दो हज़ार छब्बीस"
    for bad in ((31, 2, 2026), (0, 1, 2026), (12, 13, 2026), (1, 1, 1850)):
        with pytest.raises(ValueError):
            hi.date(*bad)


@pytest.mark.parametrize("raw, expected", [
    # money
    ("₹500", "पाँच सौ रुपये"),
    ("₹ 500", "पाँच सौ रुपये"),
    ("Rs. 500", "पाँच सौ रुपये"),
    ("Rs 500", "पाँच सौ रुपये"),
    ("rs. 500", "पाँच सौ रुपये"),
    ("रु. 500", "पाँच सौ रुपये"),
    ("₹1,50,000", "एक लाख पचास हज़ार रुपये"),
    ("₹1", "एक रुपया"),
    ("₹5.50", "पाँच रुपये पचास पैसे"),
    ("₹5 करोड़ का सौदा", "पाँच करोड़ रुपये का सौदा"),
    ("₹2.5 लाख", "ढाई लाख रुपये"),
    ("₹1.2 करोड़", "एक दशमलव दो करोड़ रुपये"),
    ("$5 का टिकट", "पाँच डॉलर का टिकट"),
    ("$5.50 में", "पाँच डॉलर पचास सेंट में"),
    ("$2 मिलियन", "दो मिलियन डॉलर"),
    ("₹1,000", "₹1,000"),                       # plain thousands grouping
    # Indian grouping, now up to crores
    ("1,50,000 लोग", "एक लाख पचास हज़ार लोग"),
    ("12,50,000 लोग", "बारह लाख पचास हज़ार लोग"),
    ("5,00,00,000 का बजट", "पाँच करोड़ का बजट"),
    ("1250000 लोग", "1250000 लोग"),             # ungrouped 7+ digits: an ID
    # fractions before a unit
    ("1.5 घंटे लगे", "डेढ़ घंटे लगे"),
    ("2.5 करोड़", "ढाई करोड़"),
    ("3.5 किलोमीटर", "साढ़े तीन किलोमीटर"),
    ("0.5 किलो", "आधा किलो"),
    ("2.25 लाख", "सवा दो लाख"),
    ("1.25 किलो", "सवा किलो"),
    ("1.75 करोड़", "पौने दो करोड़"),
    ("4.5 स्टार", "साढ़े चार स्टार"),
    ("3.7 किलोमीटर", "तीन दशमलव सात किलोमीटर"),
    ("2.5 करोड़ों", "दो दशमलव पाँच करोड़ों"),     # not the unit word itself
    ("2.5%", "ढाई प्रतिशत"),
    # dates
    ("15/08/2026 को", "पंद्रह अगस्त दो हज़ार छब्बीस को"),
    ("05-08-2026", "पाँच अगस्त दो हज़ार छब्बीस"),
    ("01.01.2000", "एक जनवरी दो हज़ार"),
    ("31/02/2026", "31/02/2026"),               # impossible date
    ("15/08/26", "15/08/26"),                   # two-digit year: ambiguous
    # mobile numbers
    ("9876543210 पर कॉल करें", "नौ आठ सात छह पाँच, चार तीन दो एक शून्य पर कॉल करें"),
    ("+91 98765 43210", "प्लस नौ एक, नौ आठ सात छह पाँच, चार तीन दो एक शून्य"),
    ("09876543210", "शून्य नौ आठ सात छह पाँच, चार तीन दो एक शून्य"),
    ("1234567890", "1234567890"),               # not a mobile number
    # clock
    ("10:30", "साढ़े दस"),
    ("10:30 बजे", "साढ़े दस बजे"),
    ("05:30 बजे", "साढ़े पाँच बजे"),
    ("10:30 बजेगा", "साढ़े दस बजेगा"),          # "बजे" inside a longer word stays
    ("24:30", "24:30"),                          # not a clock time
    # ordinals
    ("5वां", "पाँचवां"),
    ("5वाँ", "पाँचवाँ"),
    # acronyms inside Hindi
    ("CEO ने कहा", "सीईओ ने कहा"),
    ("UPI से भुगतान", "यूपीआई से भुगतान"),
    ("ISRO का मिशन", "इसरो का मिशन"),
    ("मुझे OK लगा", "मुझे ओके लगा"),
    ("I LOVE THIS", "I LOVE THIS"),             # English sentence: words, not acronyms
    ("AT&T का फ़ोन", "AT&T का फ़ोन"),
])
def test_patterns_through_the_normalizer(raw, expected):
    got = normalize_text(raw, "Hindi")
    assert unicodedata.normalize("NFC", got) == unicodedata.normalize("NFC", expected)


@pytest.mark.parametrize("unit", ["करोड़", "फ़ीसदी"])
def test_unit_words_match_in_either_nukta_spelling(unit):
    """The request path NFC-normalizes (decomposing the nukta letters); text
    from elsewhere may carry the precomposed ones. Both must match."""
    composed = unit.replace("ड़", "ड़").replace("फ़", "फ़")
    for form in {unicodedata.normalize("NFC", unit), composed}:
        assert normalize_text(f"2.5 {form}", "Hindi").startswith("ढाई"), form


def test_normalizer_output_is_idempotent_and_digit_free():
    text = (
        "डॉ. शर्मा 10:20 बजे ₹1,50,000 लेकर 5वीं मंज़िल पर 2026 में 50% छूट देंगे। "
        "CEO ने 15/08/2026 को 9876543210 पर बताया कि 2.5 करोड़ का सौदा 1.5 घंटे में हुआ।"
    )
    once = normalize_text(text, "Hindi")
    assert not any(ch.isdigit() for ch in once)
    assert normalize_text(once, "Hindi") == once
    assert unicodedata.normalize("NFC", once) == unicodedata.normalize("NFC", (
        "डॉक्टर शर्मा दस बजकर बीस मिनट पर एक लाख पचास हज़ार रुपये लेकर पाँचवीं "
        "मंज़िल पर दो हज़ार छब्बीस में पचास प्रतिशत छूट देंगे। सीईओ ने पंद्रह अगस्त "
        "दो हज़ार छब्बीस को नौ आठ सात छह पाँच, चार तीन दो एक शून्य पर बताया कि ढाई "
        "करोड़ का सौदा डेढ़ घंटे में हुआ।"
    ))
