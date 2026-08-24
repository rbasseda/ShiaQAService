"""Date-like wiki titles and infobox values -> normalised `Date` nodes.

WikiShia writes a death date as a row of separate wikilinks spanning two
calendars:

    | death/martyrdom = [[Sha'ban 15]], [[329]]/[[May 15]], [[941 CE|941]]

Four links, one date. Bare years like `[[329]]` are redirects to `329 AH` (169
of them), so the Hijri side survives link resolution — but `941 CE` is not an
article on this wiki, so the Gregorian side would be lost if we only read
resolved links. Hence both paths: resolved titles *and* a regex over the raw
value, merged into a single node with the most specific id the evidence
supports.

Hijri is the primary calendar here because it is the one the corpus indexes on
(160 `N AH` year articles against 26 `N CE`), and the system prompt requires
dates to be kept in the form the source uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Hijri months in calendar order, with the spellings this wiki actually uses as
# article titles ("Rabi' I", "Dhu l-Qa'da").
HIJRI_MONTHS: tuple[str, ...] = (
    "Muharram", "Safar", "Rabi' I", "Rabi' II", "Jumada I", "Jumada II",
    "Rajab", "Sha'ban", "Ramadan", "Shawwal", "Dhu l-Qa'da", "Dhu l-Hijja",
)
GREGORIAN_MONTHS: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _slug(text: str) -> str:
    """'Dhu l-Qa'da' -> 'dhu-l-qada'. Stable ids across apostrophe variants."""
    text = text.lower().replace("’", "'").replace("'", "")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


_HIJRI_BY_SLUG = {_slug(m): i for i, m in enumerate(HIJRI_MONTHS, 1)}
_GREG_BY_SLUG = {_slug(m): i for i, m in enumerate(GREGORIAN_MONTHS, 1)}

# "1344 AH", "941 CE", "4 BH" — as a title, or loose inside a value.
_YEAR_RE = re.compile(r"\b(\d{1,4})\s*(AH|CE|BH|AD|BC)\b", re.I)
# "Sha'ban 15", "May 15" — day always follows the month on this wiki.
_MONTH_DAY_RE = re.compile(
    r"\b(" + "|".join(
        re.escape(m) for m in
        sorted(HIJRI_MONTHS + GREGORIAN_MONTHS, key=len, reverse=True)
    ).replace("'", "['’]") + r")\s+(\d{1,2})\b",
    re.I,
)


@dataclass(frozen=True)
class DateValue:
    """A parsed date, in whichever calendars the source supplied."""

    ah_year: int | None = None
    ah_month: int | None = None
    ah_day: int | None = None
    ce_year: int | None = None
    ce_month: int | None = None
    ce_day: int | None = None
    bh_year: int | None = None   # "Before Hijra"

    def __bool__(self) -> bool:
        return any(v is not None for v in vars(self).values())

    @property
    def node_id(self) -> str:
        """Most specific stable id the evidence supports."""
        if not self:
            raise ValueError("empty DateValue has no node id; check bool(dv) first")
        if self.ah_year is not None:
            base = f"date:ah:{self.ah_year:04d}"
            if self.ah_month:
                base += f"-{self.ah_month:02d}"
                if self.ah_day:
                    base += f"-{self.ah_day:02d}"
            return base
        if self.bh_year is not None:
            return f"date:bh:{self.bh_year:04d}"
        if self.ce_year is not None:
            base = f"date:ce:{self.ce_year:04d}"
            if self.ce_month:
                base += f"-{self.ce_month:02d}"
                if self.ce_day:
                    base += f"-{self.ce_day:02d}"
            return base
        # Month and day with no year: a recurring calendar date (an anniversary
        # such as Ashura), which is a real thing on this wiki.
        if self.ah_month:
            return f"date:ah-md:{self.ah_month:02d}-{self.ah_day or 0:02d}"
        return f"date:ce-md:{self.ce_month:02d}-{self.ce_day or 0:02d}"

    @property
    def label(self) -> str:
        parts = []
        if self.ah_month:
            day = f" {self.ah_day}" if self.ah_day else ""
            parts.append(f"{HIJRI_MONTHS[self.ah_month - 1]}{day}")
        if self.ah_year is not None:
            parts.append(f"{self.ah_year} AH")
        elif self.bh_year is not None:
            parts.append(f"{self.bh_year} BH")
        hijri = ", ".join(parts)

        greg_parts = []
        if self.ce_month:
            day = f" {self.ce_day}" if self.ce_day else ""
            greg_parts.append(f"{GREGORIAN_MONTHS[self.ce_month - 1]}{day}")
        if self.ce_year is not None:
            greg_parts.append(f"{self.ce_year} CE")
        greg = ", ".join(greg_parts)

        if hijri and greg:
            return f"{hijri} / {greg}"
        return hijri or greg

    def as_node(self) -> dict:
        data = {k: v for k, v in vars(self).items() if v is not None}
        return {"id": self.node_id, "kind": "date", "label": self.label, **data}


def _merge(a: DateValue, b: DateValue) -> DateValue:
    """Combine two partial readings; the first non-None wins per component."""
    return DateValue(**{
        k: (v if v is not None else getattr(b, k)) for k, v in vars(a).items()
    })


def parse_one(text: str) -> DateValue:
    """Parse a single title or fragment ('1344 AH', "Sha'ban 15")."""
    out = DateValue()
    if m := _YEAR_RE.search(text):
        year, era = int(m.group(1)), m.group(2).upper()
        if era == "AH":
            out = _merge(out, DateValue(ah_year=year))
        elif era == "BH":
            out = _merge(out, DateValue(bh_year=year))
        else:  # CE / AD / BC
            out = _merge(out, DateValue(ce_year=year))
    if m := _MONTH_DAY_RE.search(text):
        month_slug, day = _slug(m.group(1)), int(m.group(2))
        if month_slug in _HIJRI_BY_SLUG:
            out = _merge(out, DateValue(ah_month=_HIJRI_BY_SLUG[month_slug], ah_day=day))
        elif month_slug in _GREG_BY_SLUG:
            out = _merge(out, DateValue(ce_month=_GREG_BY_SLUG[month_slug], ce_day=day))
    return out


def parse_field(raw_value: str, resolved_titles: list[str]) -> DateValue:
    """Parse one infobox date field into a single merged `DateValue`.

    `resolved_titles` are the wikilink targets that resolved to real articles
    (so `[[329]]` arrives already normalised to "329 AH"); `raw_value` is read
    as well, because the Gregorian half is often a link to a page that does not
    exist on this wiki.
    """
    out = DateValue()
    for title in resolved_titles:
        out = _merge(out, parse_one(title))
    # The raw value fills gaps only — a resolved title is the better evidence.
    return _merge(out, parse_one(raw_value))
