import asyncio
from dataclasses import replace

import pytest
from evaluation_helpers import fixture_report

from app.evaluation.loader import load_dataset
from app.evaluation.models import RunConfig, TargetOutput
from app.evaluation.runner import git_provenance, run_evaluation
from app.evaluation.targets import FixtureTarget
from app.llm.capabilities import ProviderCapabilities


def test_fixture_success_repetitions_deterministic_order_and_nonstream_latency():
    report = fixture_report(repetitions=2)
    assert report.overall.total == report.overall.successes == 58
    assert report.overall.failures == report.overall.skipped == 0
    assert report.overall.scores.quality_score == 1
    assert report.overall.latency.mean == pytest.approx(0.04)
    assert report.overall.ttft.count == 0
    assert report.overall.ttft.mean is None
    ids = [case.id for case in load_dataset().cases]
    assert [case.case_id for case in report.cases] == ids + ids
    assert [case.repetition for case in report.cases] == [1] * 29 + [2] * 29
    assert report.metadata.config.repetitions == 2


def test_fixture_reports_are_reproducible_except_run_timestamp():
    first, second = fixture_report().model_dump(), fixture_report().model_dump()
    first["metadata"].pop("generated_at")
    second["metadata"].pop("generated_at")
    assert first == second


def test_fixture_failure_count_rate_and_null_scores():
    report = fixture_report("target-b")
    assert report.overall.successes == 25
    assert report.overall.failures == 4
    assert report.overall.failure_rate == pytest.approx(4 / 29)
    assert report.overall.error_categories == {"fixture_failure": 4}
    failures = [case for case in report.cases if case.status == "failure"]
    assert all(
        all(value is None for value in case.scores.model_dump().values())
        for case in failures
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_capability_skips_are_not_failures_and_target_is_closed(streaming):
    dataset = load_dataset()
    target = FixtureTarget.load(dataset)
    target.capabilities = ProviderCapabilities(text=True, images=False, streaming=False)
    report = asyncio.run(
        run_evaluation(
            dataset, target, RunConfig(streaming=streaming), clock=target.clock
        )
    )
    assert target.closed
    assert report.overall.failures == 0
    if streaming:
        assert report.overall.skipped == 29
        assert report.overall.attempted == 0
        assert (
            report.overall.failure_rate is report.overall.scores.quality_score is None
        )
    else:
        assert report.overall.skipped == 2
        assert report.overall.attempted == 27
        assert report.overall.failure_rate == 0


def test_streaming_ttft_ignores_initial_empty_chunk_and_collects_usage():
    report = fixture_report(streaming=True, prices=True)
    assert report.overall.ttft.count == 29
    assert report.overall.ttft.mean == pytest.approx(0.01)
    assert report.overall.latency.mean == pytest.approx(0.04)
    assert report.overall.cost_available_count == 29
    assert all(
        case.input_tokens == 100 and case.output_tokens == 20 for case in report.cases
    )
    assert report.overall.scores.quality_score == 1


@pytest.mark.parametrize(("streaming", "limit"), [(False, 1), (True, 1)])
def test_output_size_limit_is_safe_failure(streaming, limit):
    dataset = load_dataset()
    target = FixtureTarget.load(dataset)
    subset = replace(dataset, cases=(dataset.cases[0],))
    report = asyncio.run(
        run_evaluation(
            subset,
            target,
            RunConfig(streaming=streaming, max_output_chars=limit),
            clock=target.clock,
        )
    )
    assert report.cases[0].status == "failure"
    assert report.cases[0].error_category == "output_limit"
    assert report.cases[0].scores.quality_score is None
    assert target.closed


def test_empty_stream_is_failure_without_ttft():
    dataset = load_dataset()
    target = FixtureTarget.load(dataset)
    case = dataset.cases[0]
    target.profile.answers[case.id] = target.profile.answers[case.id].model_copy(
        update={"content": ""}
    )
    report = asyncio.run(
        run_evaluation(
            replace(dataset, cases=(case,)),
            target,
            RunConfig(streaming=True),
            clock=target.clock,
        )
    )
    assert report.cases[0].error_category == "empty_stream"
    assert report.cases[0].ttft_seconds is None


@pytest.mark.parametrize(
    ("streaming", "after_token"), [(False, False), (True, False), (True, True)]
)
def test_cancellation_cleanup_uses_events_without_sleeps(streaming, after_token):
    async def exercise():
        dataset = load_dataset()
        entered = asyncio.Event()
        target = FixtureTarget.load(dataset)
        stream_closed = asyncio.Event()

        async def blocked_chat(case):
            entered.set()
            await asyncio.Event().wait()

        async def blocked_stream(case):
            try:
                yield TargetOutput(content="partial" if after_token else "")
                entered.set()
                await asyncio.Event().wait()
            finally:
                stream_closed.set()

        target.chat = blocked_chat
        target.stream = blocked_stream
        task = asyncio.create_task(
            run_evaluation(dataset, target, RunConfig(streaming=streaming))
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert target.closed
        if streaming:
            assert stream_closed.is_set()

    asyncio.run(exercise())


def test_attempt_timeout_is_categorized_and_target_closed():
    dataset = load_dataset()
    target = FixtureTarget.load(dataset)

    async def blocked(case):
        await asyncio.Event().wait()

    target.chat = blocked
    report = asyncio.run(
        run_evaluation(
            replace(dataset, cases=(dataset.cases[0],)),
            target,
            RunConfig(timeout_seconds=0.001),
        )
    )
    assert report.cases[0].error_category == "timeout"
    assert target.closed


def test_git_provenance_is_optional(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("TEST_SECRET_DO_NOT_LOG")

    monkeypatch.setattr("app.evaluation.runner.subprocess.run", unavailable)
    assert git_provenance() == (None, None)
