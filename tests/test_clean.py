from shiaqa.ingest.clean import clean_page, infobox_as_text

def _raw(wikitext: str, title: str = "Test Page", ns: int = 0) -> dict:
    return {"pageid": 1, "ns": ns, "title": title, "url": "http://example/Test",
            "revid": 7, "timestamp": "2025-01-01T00:00:00Z", "categories": ["Category:Testing"],
            "wikitext": wikitext}


def test_links_and_arabic_are_preserved():
    page = clean_page(_raw(
        "'''Ghadīr Khumm''' (Arabic: {{iarabic|غدير خم}}) is a pond in [[Khumm]] "
        "on the road from [[Mecca]] to [[Medina|the city]]."))
    assert "Ghadīr Khumm" in page.summary
    assert "غدير خم" in page.summary          # original script kept
    assert "Khumm" in page.summary and "the city" in page.summary  # piped link uses label
    assert "[[" not in page.summary and "'''" not in page.summary


def test_refs_leave_the_body_but_are_kept_as_citations():
    page = clean_page(_raw("He was born in Kufa.<ref>Al-Tusi, ''Rijal'', p. 12.</ref> He died there."))
    assert "<ref" not in page.summary and "Al-Tusi" not in page.summary
    assert page.citations and "Rijal" in page.citations[0]


def test_bibliography_sections_are_dropped_but_content_sections_kept():
    page = clean_page(_raw(
        "Lead text here.\n\n==Biography==\nHe studied in Najaf.\n\n"
        "==References==\n{{references}}\n\n==See Also==\n* [[Something]]\n"))
    labels = [s.label for s in page.sections]
    assert "Biography" in labels
    assert not any(x in labels for x in ("References", "See Also"))


def test_nested_subsections_get_a_path():
    page = clean_page(_raw("Lead.\n\n==Education==\nGeneral.\n\n===Medicine===\nHe studied medicine.\n"))
    assert "Education > Medicine" in [s.label for s in page.sections]


def test_infobox_becomes_structured_facts():
    page = clean_page(_raw(
        "{{Infobox Shia scholar\n | Well Known As =Hakim Ilahi\n | image =x.jpg\n"
        " | Birth =[[1267]]/1850-1\n | Epithet =\n}}\nLead text."))
    assert page.infobox["Well Known As"] == "Hakim Ilahi"
    assert page.infobox["Birth"] == "1267/1850-1"
    assert "Epithet" not in page.infobox            # blank slots dropped
    assert "Key facts about" in infobox_as_text(page.title, page.infobox)
    assert "Infobox" not in page.summary            # template did not leak into prose


def test_unbalanced_apostrophes_do_not_break_template_parsing():
    """Regression: transliterated Arabic leaves mwparserfromhell's italic state
    machine unbalanced, which used to make it miss every template on the page."""
    page = clean_page(_raw(
        "{{Infobox Shia scholar\n | Full name =al-Husayn\n"
        " | Works =''al-Qanun fi l-tibb'', ''al-Shifa''', ''Danishnama'', ...\n}}\n"
        "Ibn Sina was a philosopher. See ''al-Shifa''' for details."))
    assert page.infobox.get("Full Name") == "al-Husayn"
    assert "{{" not in page.summary and "Infobox" not in page.summary
    assert "''" not in page.summary


def test_quote_templates_keep_their_content():
    page = clean_page(_raw(
        "Lead.\n\n==Verse==\n{{pull quote|Say, 'I do not ask you any reward'.|source=Qur'an 42:23}}\n"))
    body = " ".join(s.text for s in page.sections)
    assert "I do not ask you any reward" in body
    assert "Qur'an 42:23" in body


def test_navigation_templates_and_files_vanish():
    page = clean_page(_raw(
        "{{Editorial Box\n| priority =b\n}}\n[[File:Photo.jpg|thumb|A caption]]\n"
        "{{about|X|other uses|Y}}\nReal content here.\n"))
    assert page.summary.strip() == "Real content here."


def test_main_article_pointers_are_captured_not_inlined():
    page = clean_page(_raw("Lead.\n\n==Deputies==\n{{Main|The Four Deputies}}\nThey served the Imam.\n"))
    assert "The Four Deputies" in page.pointers
    assert "Four Deputies" not in page.sections[0].text
