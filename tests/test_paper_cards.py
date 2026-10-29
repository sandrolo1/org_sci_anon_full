"""The two paper cards (paper_cards.py): the check of the quotations, the completeness measured by code, the text that is embedded, the instructions, and the selection of the
sections a card is made from. Nothing here is a real paper or a real model answer."""
from research0 import paper_cards as pc

METHOD, IMPACT = pc.METHOD, pc.IMPACT

# --- the methodology card ---------------------------------------------------------------------------------------------------------------------

SOURCE = ("We ask how enterprise social media shape accountability among employees in large firms. "
          "We surveyed 412 employees of 31 firms in two waves, six months apart, using validated multi-item scales. "
          "Accountability was measured with a four-item scale adapted from earlier work and we estimated multilevel models with random intercepts for firms. "
          "As a robustness check we repeated the analysis with firm fixed effects and with clustered standard errors.")
Q = "We ask how enterprise social media shape accountability among employees in large firms."
D = "We surveyed 412 employees of 31 firms in two waves, six months apart, using validated multi-item scales."
M = "Accountability was measured with a four-item scale adapted from earlier work"
R = "As a robustness check we repeated the analysis with firm fixed effects and with clustered standard errors."
ABSENT = "a sentence the text does not contain at all anywhere"


def field(text, status="extracted", passages=None):
    return {"text": text, "status": status, "passages": passages if passages is not None else ([] if status == "missing" else [Q])}


def missing():
    return field(None, "missing")


def card(**over):
    base = {"question": field("how enterprise social media shape accountability", passages=[Q]), "objective": field("estimate an effect", "inferred", [Q]),
            "units_levels": field("employees nested in firms", passages=[D]), "data_sample": field("412 employees of 31 firms in two waves", passages=[D]),
            "design": field("two-wave longitudinal survey", passages=[D]), "measures": field("four-item scale", passages=[M]),
            "procedure": field("multilevel models with random intercepts", passages=[M]), "assumptions_controls": field("firm fixed effects and clustered errors", passages=[R]),
            "info_gaps": []}
    base.update(over)
    return base


def test_a_valid_card_passes_unchanged_and_its_completeness_is_measured_by_code():
    fixed, demoted, problems = pc.repair_card(card(), SOURCE, METHOD)
    assert (fixed, demoted, problems) == (card(), [], [])
    sparse = card(measures=missing(), procedure=missing(), assumptions_controls=missing())
    assert pc.repair_card(sparse, SOURCE, METHOD)[0] == sparse
    done = pc.completeness_of(sparse, METHOD)
    assert done["missing"] == ["measures", "procedure", "assumptions_controls"] and done["share"] == 0.625 and done["extracted"] == 4
    assert pc.completeness_of(card(), METHOD)["share"] == 1.0


def test_a_passage_survives_the_formatting_of_a_pdf():
    ugly = SOURCE.replace("multilevel models", "multi - level  models").replace("robustness", "robust-\nness")
    assert pc.repair_card(card(), ugly, METHOD)[1:] == ([], [])


def test_the_check_demotes_a_field_it_cannot_prove_instead_of_refusing_the_card():
    bad = card(design=field("x", passages=[ABSENT]), measures=field("y", passages=[M, "another sentence the text does not contain at all"]))
    fixed, demoted, problems = pc.repair_card(bad, SOURCE, METHOD)
    assert problems == [] and demoted == ["design"]
    assert fixed["design"] == {"text": None, "status": "missing", "passages": []} and fixed["measures"]["passages"] == [M]          # the exact passage stays, the inexact one goes
    assert any(g.startswith("design: the proposed quotation") for g in fixed["info_gaps"]) and bad["design"]["status"] == "extracted"   # the input is not modified
    assert pc.completeness_of(fixed, METHOD)["missing"] == ["design"]


def test_a_card_with_fewer_than_half_its_fields_proved_is_refused_and_a_malformed_one_too():
    wrecked = card(**{f: field("x", passages=[ABSENT]) for f in METHOD.fields[:5]})
    fixed, demoted, problems = pc.repair_card(wrecked, SOURCE, METHOD)
    assert fixed is None and len(demoted) == 5 and "at least 4 are needed" in problems[0]
    exactly_half = card(**{f: field("x", passages=[ABSENT]) for f in METHOD.fields[:4]})
    assert pc.repair_card(exactly_half, SOURCE, METHOD)[0] is not None
    assert pc.repair_card("x", SOURCE, METHOD)[0] is None and pc.repair_card({**card(), "extra": 1}, SOURCE, METHOD)[0] is None


def test_a_missing_field_that_carries_text_is_cleaned_and_a_too_long_text_is_demoted():
    fixed, demoted, _ = pc.repair_card(card(procedure=field("a text", "missing", [Q]), measures=field("z" * 401, passages=[M])), SOURCE, METHOD)
    assert fixed["procedure"] == {"text": None, "status": "missing", "passages": []} and demoted == ["measures"]


def test_the_embedded_text_comes_from_the_stated_fields_in_order():
    text = pc.card_text(card(design=missing()), METHOD)
    assert text.split("\n")[0] == "how enterprise social media shape accountability" and "two-wave" not in text and len(text.split("\n")) == 7
    assert pc.card_text(card(**{f: missing() for f in METHOD.fields}), METHOD) == ""


def test_the_stored_cards_of_a_proposal_are_only_the_valid_ones():
    content = {"cards": {"method": {"status": "valid", "card": card()}, "impact": {"status": "rejected", "card": None}, "other": {"status": "valid", "card": {}}}}
    assert pc.stored_cards(content) == {"method": card()} and pc.stored_cards({"title": "old format"}) == {} and pc.stored_cards(None) == {}


def test_the_instructions_ask_for_short_continuous_quotations_in_english_and_the_versions_are_the_stored_ones():
    for system in (METHOD.card_system, IMPACT.card_system):
        assert "at most 200 characters" in system and "CONTINUOUS" in system and "footnote numbers" in system and "in ENGLISH" in system
    assert (METHOD.version, IMPACT.version) == ("anvur-method-v2", "anvur-impact-v2") and (METHOD.card_task, IMPACT.card_task) == ("anvur_method_card", "anvur_impact_card")
    schema = pc.card_schema(METHOD.fields)
    assert schema["required"] == list(METHOD.fields) + ["info_gaps"] and schema["additionalProperties"] is False


PAPER = ("Abstract\nWe study accountability.\n\n" + "Introduction\n" + "Lorem ipsum about the field. " * 400 + "\n\n"
         "2. Theory\n" + "Some theory sentences go here. " * 300 + "\n\n3. Data and Methods\n" + "We surveyed employees in two waves. " * 120 + "\n\n3.1 Measures\n" + "A four item scale was used. " * 60 +
         "\n\n4. Results\n" + "The coefficient is positive and large. " * 400 + "\n\n5. Discussion\n" + "We discuss implications. " * 200 + "\n\nReferences\n" + "Smith, J. (2020). A paper. " * 100 +
         "\n\nAppendix A\n" + "Additional robustness checks with alternative controls. " * 80 + "\n")


def test_the_method_source_keeps_the_start_and_the_method_sections_including_appendices():
    source, info = pc.method_source(PAPER)
    assert info["kind"] == "sections" and info["appendix_included"] and len(source) < 0.6 * len(PAPER)
    assert "We study accountability" in source and "We surveyed employees" in source and "Additional robustness checks" in source and "A four item scale" in source
    assert "The coefficient is positive" not in source and "We discuss implications" not in source and "Smith, J. (2020)" not in source
    assert "Data and Methods" in info["headings"] and "Appendix A" in info["headings"]


def test_without_a_method_heading_the_text_up_to_the_references_is_used():
    flat = "Plain prose without any recognisable section heading at all. " * 300 + "\nReferences\n" + "Doe, J. (2019). Cited work. " * 60
    source, info = pc.method_source(flat)
    assert info["kind"] == "full_text_fallback" and info["appendix_included"] is False and "Cited work" not in source and source.startswith("Plain prose")


def test_the_source_is_capped_and_says_so():
    source, info = pc.method_source(PAPER, max_chars=9000)
    assert len(source) <= 9000 + 20 and info["truncated"] is True


# --- the impact card --------------------------------------------------------------------------------------------------------------------------

ISOURCE = ("Prior work treats accountability as an individual trait that does not depend on the platform employees use at work. "
           "We show that enterprise social media make accountability a property of visible exchanges among colleagues. "
           "This changes how managers should design communication policies in large firms. "
           "The account applies to knowledge-intensive firms and may not extend to routine work.")
C1, C2, C3, C4 = (s if s.endswith(".") else s + "." for s in ISOURCE.split(". "))


def icard(**over):
    base = {"conversation": field("accountability at work", "inferred", [C1]), "prior_explanation": field("accountability as a trait", passages=[C1]),
            "claimed_change": field("accountability as visible exchanges", passages=[C2]), "consequences": field("redesign communication policies", passages=[C3]),
            "scope_conditions": field("knowledge-intensive firms", passages=[C4.rstrip(".")]), "info_gaps": []}
    base.update(over)
    return base


def test_the_impact_card_follows_the_same_rules_with_its_own_fields():
    assert pc.repair_card(icard(), ISOURCE, IMPACT) == (icard(), [], [])
    fixed, demoted, problems = pc.repair_card(icard(consequences=field("x", passages=[ABSENT])), ISOURCE, IMPACT)
    assert problems == [] and demoted == ["consequences"] and fixed["info_gaps"][0].startswith("consequences: the proposed quotation")
    assert pc.repair_card(icard(design=field("x")), ISOURCE, IMPACT)[0] is None                    # a field of the other card is not allowed
    done = pc.completeness_of(icard(scope_conditions=missing(), consequences=missing()), IMPACT)
    assert done["missing"] == ["consequences", "scope_conditions"] and done["share"] == 0.6 and done["extracted"] == 2         # conversation is inferred


def test_the_impact_card_has_no_methodological_field_and_asks_for_claims():
    assert not set(IMPACT.fields) & set(METHOD.fields) and "data_sample" not in IMPACT.card_system and "measures" not in IMPACT.card_system
    assert "do not evaluate" in IMPACT.card_system and "CLAIMS" in IMPACT.card_system


IPAPER = ("Abstract\nWe study accountability.\n\nIntroduction\n" + "Introductory sentences about the field. " * 300 + "We contribute by showing a new mechanism. " * 5 +
          "\n\n2. Theory\n" + "Some theory sentences go here. " * 300 + "\n\n3. Methods\n" + "We surveyed employees in two waves. " * 200 + "\n\n4. Results\n" + "The coefficient is positive and large. " * 300 +
          "\n\n5. Discussion\n" + "The findings change how we understand accountability at work. " * 100 + "\n\n5.1 Practical implications\n" + "Managers should redesign policies. " * 60 +
          "\n\n5.2 Limitations and future research\n" + "A limitation is the single country. " * 40 + "\n\nReferences\n" + "Smith, J. (2020). A paper. " * 100 + "\n")


def test_the_impact_source_keeps_the_start_and_the_sections_that_state_a_contribution():
    source, info = pc.impact_source(IPAPER)
    assert info["kind"] == "sections" and len(source) < 0.45 * len(IPAPER)
    assert "We study accountability" in source and "change how we understand accountability" in source and "Managers should redesign" in source and "single country" in source
    assert "We surveyed employees" not in source and "The coefficient is positive" not in source and "Smith, J. (2020)" not in source


def test_results_and_discussion_counts_as_discussion_but_plain_results_do_not():
    text = "Abstract\n" + "x " * 3000 + "\n\n4. Results and Discussion\n" + "The change we describe is large. " * 120 + "\n\nReferences\n" + "Doe 2019. " * 50
    source, info = pc.impact_source(text)
    assert info["kind"] == "sections" and "The change we describe is large" in source


def test_without_a_section_the_impact_source_is_the_start_and_the_end_before_the_references():
    flat = "Opening prose of the paper. " * 500 + "Closing prose where the authors state what changes. " * 600 + "\nReferences\n" + "Doe, J. (2019). Cited. " * 60
    source, info = pc.impact_source(flat)
    assert info["kind"] == "full_text_fallback" and info["truncated"] and source.startswith("Opening prose") and source.rstrip().endswith("what changes.") and "Cited" not in source
