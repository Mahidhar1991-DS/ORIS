from dataclasses import dataclass, field

from app.models import ModelCall

# GPT-5.4 mini:

MODEL_PRICING = {
    "gpt-5.4-mini": {
        "input_per_million": 0.75,
        "output_per_million": 4.50,
    },
}


def calculate_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float:
    """
    Calculate the estimated model cost in USD.

    Returns 0.0 when the model is unknown rather than
    inventing a price.
    """

    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        return 0.0

    input_cost = (
        tokens_in
        / 1_000_000
        * pricing["input_per_million"]
    )

    output_cost = (
        tokens_out
        / 1_000_000
        * pricing["output_per_million"]
    )

    return round(
        input_cost + output_cost,
        8,
    )


def create_model_call(
    model: str,
    purpose: str,
    tokens_in: int,
    tokens_out: int,
) -> ModelCall:
    """
    Create a ModelCall containing token usage and cost.
    """

    cost = calculate_cost(
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    return ModelCall(
        model=model,
        purpose=purpose,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        usd_cost=cost,
    )


@dataclass
class CostTracker:
    max_usd_cost: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    usd_cost: float = 0.0
    model_calls: list[ModelCall] = field(
        default_factory=list
    )

    def can_call(
        self,
        estimated_cost: float = 0.0,
    ) -> bool:
        """
        Return True when another call is allowed.

        If no budget is configured, calls are allowed.
        """

        if self.max_usd_cost is None:
            return True

        return (
            self.usd_cost + estimated_cost
            <= self.max_usd_cost
        )

    def record(
        self,
        model_call: ModelCall,
    ) -> None:
        """
        Add one completed model call to the tracker.
        """

        self.tokens_in += model_call.tokens_in
        self.tokens_out += model_call.tokens_out
        self.usd_cost += model_call.usd_cost

        self.usd_cost = round(
            self.usd_cost,
            8,
        )

        self.model_calls.append(
            model_call
        )

    @property
    def remaining_usd(self) -> float | None:
        """
        Return remaining budget.

        Returns None when no budget is configured.
        """

        if self.max_usd_cost is None:
            return None

        return max(
            0.0,
            round(
                self.max_usd_cost
                - self.usd_cost,
                8,
            ),
        )
