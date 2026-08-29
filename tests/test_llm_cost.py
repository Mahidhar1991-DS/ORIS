import pytest

from app.llm_cost import (
    CostTracker,
    calculate_cost,
    create_model_call,
)


def test_calculate_cost():
    cost = calculate_cost(
        "gpt-5.4-mini",
        1_000_000,
        1_000_000,
    )

    assert cost == pytest.approx(5.25)


def test_create_model_call():
    call = create_model_call(
        model="gpt-5.4-mini",
        purpose="extraction",
        tokens_in=1_000,
        tokens_out=200,
    )

    assert call.model == "gpt-5.4-mini"
    assert call.purpose == "extraction"
    assert call.tokens_in == 1_000
    assert call.tokens_out == 200
    assert call.usd_cost > 0


def test_cost_tracker_records_call():
    tracker = CostTracker(
        max_usd_cost=15.0,
    )

    call = create_model_call(
        model="gpt-5.4-mini",
        purpose="extraction",
        tokens_in=1_000,
        tokens_out=200,
    )

    tracker.record(call)

    assert tracker.tokens_in == 1_000
    assert tracker.tokens_out == 200
    assert tracker.usd_cost == pytest.approx(
        call.usd_cost
    )
    assert len(tracker.model_calls) == 1
    assert tracker.model_calls[0] == call


def test_cost_tracker_allows_call_under_budget():
    tracker = CostTracker(
        max_usd_cost=15.0,
    )

    assert tracker.can_call() is True


def test_cost_tracker_stops_after_budget():
    tracker = CostTracker(
        max_usd_cost=0.001,
    )

    call = create_model_call(
        model="gpt-5.4-mini",
        purpose="extraction",
        tokens_in=1_000_000,
        tokens_out=1_000_000,
    )

    tracker.record(call)

    assert tracker.usd_cost > 0.001
    assert tracker.can_call() is False

def test_cost_tracker_rejects_estimated_call_over_budget():
    tracker = CostTracker(
        max_usd_cost=0.001,
    )

    assert tracker.can_call(
        estimated_cost=0.0011
    ) is False

def test_cost_tracker_allows_estimated_call_under_budget():
    tracker = CostTracker(
        max_usd_cost=0.001,
    )

    assert tracker.can_call(
        estimated_cost=0.0009
    ) is True