from __future__ import annotations

import asyncio
import platform
import subprocess
import time
from collections.abc import Callable
from contextlib import aclosing
from datetime import UTC, datetime

from app.core.observability import error_category
from app.evaluation.loader import PROJECT_ROOT, Dataset
from app.evaluation.models import (
    CaseResult,
    EvaluationCase,
    EvaluationReport,
    Provenance,
    RunConfig,
    TargetOutput,
)
from app.evaluation.pricing import PriceCatalog, compute_cost
from app.evaluation.reporting import build_report
from app.evaluation.scoring import score_case
from app.evaluation.targets import EvaluationTarget, TargetFailure


def git_provenance() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
        return sha, bool(dirty)
    except (OSError, subprocess.SubprocessError):
        return None, None


async def evaluate_case(
    dataset: Dataset,
    case: EvaluationCase,
    target: EvaluationTarget,
    repetition: int,
    config: RunConfig,
    prices: PriceCatalog | None,
    clock: Callable[[], float],
) -> CaseResult:
    base = dict(
        case_id=case.id,
        domain=case.domain,
        language=case.language,
        repetition=repetition,
        weight=case.weight,
        provider=target.provider,
        model=target.model,
    )
    capabilities = target.capabilities
    if (
        (case.requirements.text and not capabilities.text)
        or (case.requirements.images and not capabilities.images)
        or (
            (config.streaming or case.requirements.streaming)
            and not capabilities.streaming
        )
    ):
        return CaseResult.model_validate(
            {**base, "status": "skipped", "skip_reason": "capability"}
        )
    started = clock()
    ttft = None
    output = TargetOutput(content="")
    chunks: list[str] = []
    length = 0
    try:
        async with asyncio.timeout(config.timeout_seconds):
            if config.streaming:
                stream = target.stream(case)
                async with aclosing(stream):
                    async for chunk in stream:
                        if (
                            chunk.usage_reliable
                            and chunk.input_tokens is not None
                            and chunk.output_tokens is not None
                        ):
                            output = chunk
                        if not chunk.content:
                            continue
                        if ttft is None:
                            ttft = max(0.0, clock() - started)
                        length += len(chunk.content)
                        if length > config.max_output_chars:
                            raise TargetFailure("output_limit")
                        chunks.append(chunk.content)
                if ttft is None:
                    raise TargetFailure("empty_stream")
                output = output.model_copy(update={"content": "".join(chunks)})
            else:
                output = await target.chat(case)
                length = len(output.content)
                if length > config.max_output_chars:
                    raise TargetFailure("output_limit")
    except Exception as error:
        category = (
            error.category
            if isinstance(error, TargetFailure)
            else (
                "timeout" if isinstance(error, TimeoutError) else error_category(error)
            )
        )
        return CaseResult.model_validate(
            {
                **base,
                "status": "failure",
                "error_category": category,
                "latency_seconds": max(0.0, clock() - started),
                "ttft_seconds": ttft,
                "output_chars": length,
            }
        )
    latency = max(0.0, clock() - started)
    cost = compute_cost(target.provider, target.model, output, prices)
    return CaseResult.model_validate(
        {
            **base,
            "status": "success",
            "latency_seconds": latency,
            "ttft_seconds": ttft,
            "output_chars": len(output.content),
            "scores": score_case(
                case, output.content, dataset.manifest.quality_weights
            ),
            "input_tokens": output.input_tokens if output.usage_reliable else None,
            "output_tokens": output.output_tokens if output.usage_reliable else None,
            "cost_usd": cost,
            "cost_status": "computed" if cost is not None else "unavailable",
        }
    )


async def run_evaluation(
    dataset: Dataset,
    target: EvaluationTarget,
    config: RunConfig,
    *,
    prices: PriceCatalog | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> EvaluationReport:
    results = []
    try:
        for repetition in range(1, config.repetitions + 1):
            for case in dataset.cases:
                results.append(
                    await evaluate_case(
                        dataset, case, target, repetition, config, prices, clock
                    )
                )
    finally:
        await target.close()
    sha, dirty = git_provenance()
    metadata = Provenance(
        dataset_version=dataset.manifest.dataset_version,
        dataset_hash=dataset.hash,
        scorer_config_hash=dataset.scorer_hash,
        price_catalog_hash=prices.hash if prices else None,
        asset_mapping_hash=target.asset_mapping_hash,
        fixture_profile_hash=target.profile_hash,
        git_sha=sha,
        git_dirty=dirty,
        python_version=platform.python_version(),
        generated_at=datetime.now(UTC).isoformat(),
        provider=target.provider,
        model=target.model,
        mode="fixture" if target.fixture else "live",
        case_ids=[case.id for case in dataset.cases],
        config=config,
    )
    return build_report(metadata, results)
