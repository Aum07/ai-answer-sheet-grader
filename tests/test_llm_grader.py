"""LLM grader tests with a fake OpenAI client (no network, no API key)."""

import json
from types import SimpleNamespace

from grader.llm_grader import LLMGrader


class FakeClient:
    """Mimics client.chat.completions.create; scores = answer length // 10, can skip ids once."""

    def __init__(self, skip_first: set[str] = frozenset()):
        self.calls = 0
        self.skip = set(skip_first)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, temperature, response_format, messages):
        self.calls += 1
        payload = json.loads(messages[1]["content"])
        grades = []
        for a in payload["student_answers"]:
            if a["id"] in self.skip:
                self.skip.discard(a["id"])
                continue
            grades.append({"id": a["id"], "score": min(10, len(a["answer"]) // 10), "feedback": f"fb {a['id']}"})
        msg = SimpleNamespace(content=json.dumps({"grades": grades}))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def test_batches_one_call_per_question_and_keeps_order(tmp_path):
    client = FakeClient()
    g = LLMGrader(client=client, cache_dir=tmp_path)
    answers = ["short", "a much longer answer here", ""]
    grades = g.grade_question("Q?", "Ref.", answers)
    assert [x.score for x in grades] == [0, 2, 0]
    assert grades[2].feedback == "No answer was given."  # blank never sent to the API
    assert client.calls == 1
    assert g.usage.cost_usd > 0


def test_cache_avoids_paying_twice(tmp_path):
    client = FakeClient()
    LLMGrader(client=client, cache_dir=tmp_path).grade_question("Q?", "Ref.", ["answer one", "answer two"])
    g2 = LLMGrader(client=client, cache_dir=tmp_path)
    g2.grade_question("Q?", "Ref.", ["answer one", "answer two"])
    assert client.calls == 1 and g2.usage.cached == 2


def test_retries_only_skipped_ids(tmp_path):
    client = FakeClient(skip_first={"2"})
    grades = LLMGrader(client=client, cache_dir=tmp_path).grade_question("Q?", "Ref.", ["first answer", "second answer"])
    assert client.calls == 2
    assert [x.feedback for x in grades] == ["fb 1", "fb 2"]  # second call asked only for id 2
