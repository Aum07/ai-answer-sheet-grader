from grader.adversarial import ATTACKS, FILLER, keywords, keyword_stuff, pad, repeat, truncate

REF = "A stack is a last-in first-out data structure supporting push and pop operations in constant time."


def test_keywords_are_content_words_in_order():
    kw = keywords(REF)
    assert kw[:3] == ["stack", "last-in", "first-out"]
    assert "the" not in kw and "and" not in kw


def test_perturbations():
    ans = "A stack is LIFO and supports push and pop."
    assert pad(ans).endswith(FILLER) and len(pad(ans)) > 2 * len(ans)
    assert repeat(ans) == f"{ans} {ans}"
    assert truncate("one two three four") == "one two"
    assert "." not in keyword_stuff(ans, reference=REF)


def test_every_attack_has_expectation():
    assert {e for _, e, _ in ATTACKS.values()} <= {"increase", "decrease", "no increase"}
