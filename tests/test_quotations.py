"""Quotation marks in an answer's own text, checked against the passages it cites."""

from verbatim.quotations import QuotationProblem, check_quotations

FLUX = "## Intro\n\nMagnetic flux threads the ring. The current decays when the junction warms."
QUBIT = "Superconducting qubits decohere quickly at higher temperatures."


def test_words_found_in_their_passage_keep_their_quotation_marks() -> None:
    text = 'The authors write that "the current decays when the junction warms" [1].'

    assert check_quotations(text, [FLUX]) == (text, [])


def test_words_the_passage_does_not_hold_lose_their_quotation_marks() -> None:
    text = 'They report that "the current decays whenever the junction warms" [1].'

    checked, problems = check_quotations(text, [FLUX])

    assert checked == "They report that the current decays whenever the junction warms [1]."
    (problem,) = problems
    assert problem.quote == "the current decays whenever the junction warms"
    assert (problem.marker, problem.found_in) == (1, None)


def test_a_changed_word_is_not_forgiven() -> None:
    """The model's usual slip: a quote bent to fit its sentence, "decays" made "decay"."""
    text = 'Currents "decay when the junction warms" [1].'

    checked, problems = check_quotations(text, [FLUX])

    assert checked == "Currents decay when the junction warms [1]."
    assert len(problems) == 1


def test_words_from_another_cited_passage_keep_their_marks_and_are_reported() -> None:
    text = 'Qubits "decohere quickly at higher temperatures" [1].'

    checked, problems = check_quotations(text, [FLUX, QUBIT])

    assert checked == text
    (problem,) = problems
    assert (problem.marker, problem.found_in) == (1, 2)
    assert "[2], not [1]" in problem.describe()


def test_what_the_citation_check_forgives_is_forgiven_here_too() -> None:
    """Spacing, trailing punctuation, and a capital first letter dropped mid-sentence."""
    text = (
        'They note "the current   decays when the junction warms." [1] '
        'and that "magnetic flux threads the ring" [1].'
    )

    assert check_quotations(text, [FLUX]) == (text, [])


def test_a_short_quoted_term_is_left_alone() -> None:
    """Two words in quotation marks are usually a term, not a quotation."""
    text = 'The "lost middle" effect matters here [1].'

    assert check_quotations(text, [FLUX]) == (text, [])


def test_curly_quotation_marks_are_checked_too() -> None:
    kept = "They write “the current decays when the junction warms” [1]."
    bent = "They write “the current fades when the junction warms” [1]."

    assert check_quotations(kept, [FLUX]) == (kept, [])
    assert check_quotations(bent, [FLUX])[0] == "They write the current fades when the junction warms [1]."


def test_a_quotation_inside_a_quotation_is_part_of_it() -> None:
    """A real answer quoted a passage that itself quotes, and a naive pairing split it."""
    passage = "Binaries split into first-generation “1G+1G” mergers and later ones."
    text = 'They are grouped into "first-generation “1G+1G” mergers" [1]. Masses "can be measured directly" [1].'

    checked, problems = check_quotations(text, [passage])

    assert [problem.quote for problem in problems] == ["can be measured directly"]
    assert checked.startswith('They are grouped into "first-generation “1G+1G” mergers" [1].')
    assert checked.endswith("Masses can be measured directly [1].")


def test_an_odd_number_of_straight_marks_leaves_them_alone_rather_than_guess() -> None:
    text = 'A 3.5" disk, and "words that no passage holds" [1].'

    assert check_quotations(text, [FLUX]) == (text, [])


def test_a_phrase_with_no_marker_in_its_sentence_is_checked_against_every_cited_passage() -> None:
    found = 'Qubits "decohere quickly at higher temperatures" here. Flux matters too [1].'
    missing = 'Qubits "decohere slowly at any temperature" here. Flux matters too [1].'

    assert check_quotations(found, [FLUX, QUBIT]) == (found, [])
    checked, problems = check_quotations(missing, [FLUX, QUBIT])
    assert checked.startswith("Qubits decohere slowly at any temperature here.")
    assert problems[0].marker is None


def test_a_group_of_markers_accepts_any_of_them() -> None:
    text = 'Qubits "decohere quickly at higher temperatures" [1][2].'

    assert check_quotations(text, [FLUX, QUBIT]) == (text, [])


def test_an_unverified_marker_names_no_passage() -> None:
    text = 'They claim "words that no passage holds" [unverified].'

    checked, problems = check_quotations(text, [FLUX])

    assert checked == "They claim words that no passage holds [unverified]."
    assert problems[0].marker is None


def test_with_nothing_cited_no_quotation_can_stand() -> None:
    text = 'They claim "words that no passage holds" [unverified].'

    assert check_quotations(text, [])[0] == "They claim words that no passage holds [unverified]."


def test_a_problem_says_what_was_done() -> None:
    removed = QuotationProblem(quote="a b c", marker=1, found_in=None)
    elsewhere = QuotationProblem(quote="d e f", marker=1, found_in=3)

    assert "without quotation marks" in removed.describe()
    assert elsewhere.describe() == "quoted from [3], not [1]"
