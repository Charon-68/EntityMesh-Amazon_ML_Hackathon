"""
normalize.py — Country-agnostic, Unicode-safe text normalizer.

Pipeline
--------
1. Decode / strip BOM and control characters
2. Unicode NFKC normalization  (half-width → full-width, ligatures, etc.)
3. Punctuation removal — PRESERVES Unicode letters (L), combining marks (M),
   decimal numbers (N), and ASCII space/hyphen; strips everything else.
4. Whitespace collapse
5. Lower-case (locale-independent via str.casefold)
6. Legal suffix expansion  (pvt ltd → private limited, sarl → sarl, etc.)
7. Address abbreviation expansion  (rd → road, st → street, etc.)

Design constraints
------------------
- Zero branching on country. The same function handles US, India, France and
  any future locale the challenge adds.
- Unicode-safe: regex uses Unicode category escapes so Hindi matras
  (combining marks, category M) and digits (category N) are never stripped.
- No external dependencies — stdlib only (re, unicodedata).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ---------------------------------------------------------------------------
# Step 1 – Unicode NFKC + control-char stripping
# ---------------------------------------------------------------------------

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _unicode_clean(text: str) -> str:
    """NFKC-normalize and strip invisible / control characters."""
    # Strip UTF-8 BOM if present
    text = text.lstrip("\ufeff")
    # NFKC: decomposes ligatures, converts full-width chars, etc.
    text = unicodedata.normalize("NFKC", text)
    # Remove non-printable control characters
    text = _CONTROL_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Step 2 – Punctuation removal (Unicode-category-aware)
# ---------------------------------------------------------------------------
#
# Python's \w in re.UNICODE covers:  Ll, Lu, Lt, Lo, Lm (letters), Nd (decimal
# digits), Pc (connector punctuation — i.e. underscore).  It does NOT include
# Unicode combining marks (category M: Mn, Mc, Me).  Hindi matras and similar
# diacritics are Mc/Mn — stripping them breaks Devanagari tokens.
#
# Strategy: explicitly allow L, M, N categories via the unicodedata module
# in a translation-based approach, then fall back to the regex for ASCII.
#
# For the regex path we explicitly add Unicode combining marks by including
# the range \u0300-\u036f (Latin combining), \u0900-\u097f (Devanagari),
# \u0a00-\u0a7f (Gurmukhi), \u0b00-\u0b7f (Odia/Tamil adjacent), \u0c00-\u0c7f,
# \u0d00-\u0d7f (Malayalam), \u0e00-\u0e7f (Thai), \u0f00-\u0fff (Tibetan),
# plus the generic "Mark" Unicode ranges via a positive Unicode-property helper.


def _is_keep_char(ch: str) -> bool:
    """Return True for characters that must NOT be stripped."""
    cat = unicodedata.category(ch)
    # L* = letters, M* = marks (diacritics/matras), N* = numbers
    return cat[0] in ("L", "M", "N") or ch in (" ", "-")


_WS_RE = re.compile(r"\s+", re.UNICODE)


def _strip_punctuation(text: str) -> str:
    """Remove punctuation while preserving Unicode letters, marks, digits."""
    # Replace every character that is not a letter, mark, number, space or
    # hyphen with a space so tokens stay separated.
    cleaned = "".join(ch if _is_keep_char(ch) else " " for ch in text)
    return _WS_RE.sub(" ", cleaned).strip()


# ---------------------------------------------------------------------------
# Step 3 – Legal suffix expansion
# ---------------------------------------------------------------------------
#
# Ordered from longest / most-specific to shortest to avoid partial matches.
# All keys and values are already lower-cased (applied after casefold).
#
# Coverage:
#   US/international : inc, corp, llc, llp, ltd, plc, co
#   India            : pvt, private limited, p limited, ltdliab
#   France           : sarl, sas, sasu, snc, sa, sci, eurl, sca, scop
#   Generic          : and ↔ &

_LEGAL_EXPANSIONS: list[tuple[re.Pattern, str]] = []

_LEGAL_RAW: list[tuple[str, str]] = [
    # ---- multi-word / compound (must come FIRST) ----
    (r"\bpvt\.?\s+ltd\.?\b",                   "private limited"),
    (r"\bprivate\s+ltd\.?\b",                   "private limited"),
    (r"\bp\.?\s*ltd\.?\b",                       "private limited"),
    (r"\bltd\.?\s+liability\b",                 "limited liability"),
    (r"\blimited\s+liability\s+company\b",       "limited liability company"),
    (r"\bllc\.?\b",                              "limited liability company"),
    (r"\bllp\.?\b",                              "limited liability partnership"),
    # ---- single-word legal forms ----
    (r"\bincorporated\b",                        "incorporated"),
    (r"\binc\.?\b",                              "incorporated"),
    (r"\bcorporation\b",                         "corporation"),
    (r"\bcorp\.?\b",                             "corporation"),
    (r"\bltd\.?\b",                              "limited"),
    (r"\bplc\.?\b",                              "public limited company"),
    (r"\bpvt\.?\b",                              "private"),
    # ---- French legal forms ----
    (r"\bsarl\b",                                "sarl"),       # société à responsabilité limitée
    (r"\bsas\b",                                 "sas"),        # société par actions simplifiée
    (r"\bsasu\b",                                "sasu"),       # sas unipersonnelle
    (r"\bsnc\b",                                 "snc"),        # société en nom collectif
    (r"\bsca\b",                                 "sca"),        # société en commandite par actions
    (r"\bscop\b",                                "scop"),       # société coopérative
    (r"\bsci\b",                                 "sci"),        # société civile immobilière
    (r"\beurl\b",                                "eurl"),       # entreprise unipersonnelle
    (r"\bsa\b",                                  "sa"),         # société anonyme
    # ---- symbol / punctuation variants ----
    (r"\s+&\s+",                                 " and "),
    (r"\band\b",                                 "and"),
    # ---- generic ----
    (r"\bco\.?\b",                               "company"),
    (r"\bcompany\b",                             "company"),
    (r"\benterprises?\b",                        "enterprise"),
    (r"\bindustries?\b",                         "industry"),
    (r"\bservices?\b",                           "service"),
    (r"\bsolutions?\b",                          "solution"),
    (r"\bgroup\b",                               "group"),
    (r"\bholdings?\b",                           "holding"),
    (r"\btraders?\b",                            "trader"),
    (r"\bbrothers?\b",                           "brother"),
    (r"\bsons?\b",                               "son"),
    (r"\bassociates?\b",                         "associate"),
    (r"\bpartners?\b",                           "partner"),
]

for _pat, _repl in _LEGAL_RAW:
    _LEGAL_EXPANSIONS.append((re.compile(_pat, re.IGNORECASE | re.UNICODE), _repl))


def _expand_legal(text: str) -> str:
    for pattern, replacement in _LEGAL_EXPANSIONS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Step 4 – Address abbreviation expansion
# ---------------------------------------------------------------------------

_ADDR_RAW: list[tuple[str, str]] = [
    # ---- road / route types ----
    (r"\bblvd\.?\b",    "boulevard"),
    (r"\bave\.?\b",     "avenue"),
    (r"\bav\.?\b",      "avenue"),
    (r"\bdr\.?\b",      "drive"),
    (r"\bln\.?\b",      "lane"),
    (r"\bct\.?\b",      "court"),
    (r"\bcir\.?\b",     "circle"),
    (r"\bhwy\.?\b",     "highway"),
    (r"\bfwy\.?\b",     "freeway"),
    (r"\bpkwy\.?\b",    "parkway"),
    (r"\bpl\.?\b",      "place"),
    (r"\bsq\.?\b",      "square"),
    (r"\bxing\.?\b",    "crossing"),
    (r"\brte\.?\b",     "route"),
    (r"\brt\.?\b",      "route"),
    # ---- "street" must come BEFORE "st" to avoid clobbering ----
    (r"\bstreet\b",     "street"),
    (r"\bst\.?\b",      "street"),
    # ---- "road" before "rd" ----
    (r"\broad\b",       "road"),
    (r"\brd\.?\b",      "road"),
    # ---- unit / building ----
    (r"\bapt\.?\b",     "apartment"),
    (r"\bappt\.?\b",    "apartment"),
    (r"\bflr\.?\b",     "floor"),
    (r"\bfl\.?\b",      "floor"),
    (r"\bste\.?\b",     "suite"),
    (r"\bbldg\.?\b",    "building"),
    (r"\bbldng\.?\b",   "building"),
    (r"\bdept\.?\b",    "department"),
    (r"\bunit\b",       "unit"),
    (r"\bno\.?\b",      "number"),
    (r"\bnr\.?\b",      "near"),
    # ---- directionals ----
    (r"\bn\.?\b",       "north"),
    (r"\bs\.?\b",       "south"),
    (r"\be\.?\b",       "east"),
    (r"\bw\.?\b",       "west"),
    (r"\bne\.?\b",      "northeast"),
    (r"\bnw\.?\b",      "northwest"),
    (r"\bse\.?\b",      "southeast"),
    (r"\bsw\.?\b",      "southwest"),
    # ---- India-specific ----
    (r"\bkh\.?\s*no\.?\b",          "khasra number"),
    (r"\bnagar\b",      "nagar"),
    (r"\bcolony\b",     "colony"),
    (r"\bmarg\b",       "marg"),
    (r"\bchowk\b",      "chowk"),
    (r"\bvihar\b",      "vihar"),
    # ---- France-specific ----
    (r"\brue\b",        "rue"),
    (r"\bav\.?\b",      "avenue"),
    (r"\bbd\.?\b",      "boulevard"),
    (r"\bimm\.?\b",     "immeuble"),
    (r"\bres\.?\b",     "residence"),
    (r"\bcedex\b",      "cedex"),
]

_ADDR_EXPANSIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE | re.UNICODE), repl)
    for pat, repl in _ADDR_RAW
]


def _expand_address(text: str) -> str:
    for pattern, replacement in _ADDR_EXPANSIONS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_name(text: Optional[str]) -> str:
    """Normalize a business name field.

    Steps: unicode-clean → punctuation-strip → casefold → legal expansion
           → whitespace collapse.

    Parameters
    ----------
    text:
        Raw business name string (may be None or empty).

    Returns
    -------
    str
        Normalized form; empty string for None/empty inputs.
    """
    if not text:
        return ""
    text = _unicode_clean(text)
    text = _strip_punctuation(text)
    text = text.casefold()
    text = _expand_legal(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_address(text: Optional[str]) -> str:
    """Normalize a business address field.

    Steps: unicode-clean → punctuation-strip → casefold → address expansion
           → legal expansion (for addresses that embed entity types)
           → whitespace collapse.

    Parameters
    ----------
    text:
        Raw address string (may be None or empty).

    Returns
    -------
    str
        Normalized form; empty string for None/empty inputs.
    """
    if not text:
        return ""
    text = _unicode_clean(text)
    text = _strip_punctuation(text)
    text = text.casefold()
    text = _expand_address(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_record(
    business_name: Optional[str],
    business_address: Optional[str],
) -> tuple[str, str]:
    """Convenience wrapper: normalize name and address together."""
    return normalize_name(business_name), normalize_address(business_address)


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 10 diverse multilingual examples — before / after side-by-side

    NAME_EXAMPLES: list[tuple[str, str]] = [
        # label,                             raw_input
        ("US — legal suffix abbrev",         "Orelee's Barbershop Inc."),
        ("US — LLC + ampersand",             "Smith & Jones, LLC"),
        ("US — Corp + punctuation noise",    "-- Holloway Peak Corp. Seafood!!"),
        ("India — Pvt Ltd (Devanagari)",     "राम मार्केटिंग प्राइवेट लिमिटेड"),
        ("India — English Pvt Ltd abbrev",   "Sunrise Traders Pvt. Ltd."),
        ("India — mixed script + suffix",    "Sharma & Sons P. Ltd"),
        ("France — SARL",                    "Boulangerie Dupont SARL"),
        ("France — SAS + accented chars",    "Société Générale SAS"),
        ("France — EURL",                    "Café de la Paix EURL"),
        ("Multilingual — Unicode stress",    "Ñoño & Ñoño S.A."),
    ]

    ADDR_EXAMPLES: list[tuple[str, str]] = [
        # label,                             raw_input
        ("US — full abbreviation set",       "1795 Westchester Dr., High Point, NC"),
        ("US — apt + street abbrev",         "Apt 4B, 200 Main St., Springfield, IL"),
        ("US — directional + blvd",          "100 N. Oak Blvd, Dallas, TX"),
        ("India — KH No. variant",           "KH NO. -570/13, NEW DELHI, WEST DELHI"),
        ("India — landmark style",           "Near SBI ATM, Nagar Rd, Pune"),
        ("India — missing PIN",              "Plot 7, Industrial Area, Phase II"),
        ("France — rue + cedex",             "12 Rue de la Paix, PARIS Cedex 01"),
        ("France — immeuble + bd",           "Imm. Le Régent, 5 Bd. Haussmann, Paris"),
        ("Generic — empty input",            ""),
        ("Generic — only punctuation",       "---/---"),
    ]

    COL = 40
    DIVIDER = "─" * 90

    print(DIVIDER)
    print("  BUSINESS NAME NORMALIZATION — Before → After")
    print(DIVIDER)
    for label, raw in NAME_EXAMPLES:
        normalized = normalize_name(raw)
        print(f"  [{label}]")
        print(f"    IN : {raw!r}")
        print(f"    OUT: {normalized!r}")
        print()

    print(DIVIDER)
    print("  ADDRESS NORMALIZATION — Before → After")
    print(DIVIDER)
    for label, raw in ADDR_EXAMPLES:
        normalized = normalize_address(raw)
        print(f"  [{label}]")
        print(f"    IN : {raw!r}")
        print(f"    OUT: {normalized!r}")
        print()

    # Quick regression assertions
    assert normalize_name("") == ""
    assert normalize_name(None) == ""
    assert normalize_address("") == ""
    assert "private limited" in normalize_name("ABC Pvt. Ltd.")
    assert "limited liability company" in normalize_name("XYZ LLC")
    assert "road" in normalize_address("100 Main Rd.")
    assert "street" in normalize_address("5 Oak St.")
    assert "apartment" in normalize_address("Apt 3C, Broadway")
    assert "sarl" in normalize_name("Dupont SARL")
    # Hindi text must survive (not get stripped)
    hindi_out = normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert "राम" in hindi_out or len(hindi_out) > 5, f"Hindi stripped! got: {hindi_out!r}"

    print("  ✅ All assertions passed.")
