from time import monotonic
from collections import defaultdict
from fastapi import HTTPException


class TokenBucketRateLimiter:

    def __init__(self, capacity: int = 10, refill_rate: float = 1.0) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = defaultdict(lambda: float(capacity))
        self._last_refill = defaultdict(monotonic)

    def _refill(self, user_id: str) -> None:
        last_refill_time = self._last_refill[user_id]
        current_time = monotonic()
        elapsed_time = current_time - last_refill_time
        self._tokens[user_id] = min(
            self.capacity,
            self._tokens[user_id] + elapsed_time * self.refill_rate)
        self._last_refill[user_id] = current_time

    def check(self, user_id: str) -> None:

        self._refill(user_id)
        if self._tokens[user_id] < 1:
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")
        self._tokens[user_id] -= 1
