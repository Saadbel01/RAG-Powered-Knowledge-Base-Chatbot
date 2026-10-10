from time import monotonic
from collections import defaultdict
from fastapi import HTTPException
import redis
from functools import lru_cache
from rag_chatbot.config import get_settings


_LUA_TOKEN_BUCKET = """
    local key         = KEYS[1]
    local capacity    = tonumber(ARGV[1])
    local rate        = tonumber(ARGV[2])
    local now         = tonumber(ARGV[3])

    local data        = redis.call("HMGET", key, "tokens", "last_refill")
    local tokens      = tonumber(data[1]) or capacity
    local last_refill = tonumber(data[2]) or now

    local elapsed = now - last_refill
    tokens = math.min(capacity, tokens + elapsed * rate)

    if tokens < 1 then
        redis.call("HSET", key, "tokens", tokens, "last_refill", now)
        redis.call("EXPIRE", key, 3600)
        return 0
    end

    tokens = tokens - 1
    redis.call("HSET", key, "tokens", tokens, "last_refill", now)
    redis.call("EXPIRE", key, 3600)
    return 1
"""


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


@lru_cache(maxsize=1)
def _get_redis() -> redis.Redis:
    cfg = get_settings()
    return redis.from_url(cfg.redis_url, decode_responses=True)


def check_rate_limit(user_id: str, capacity: int = 10,
                     refill_rate: float = 1.0) -> None:

    r = _get_redis()
    script = r.register_script(_LUA_TOKEN_BUCKET)
    key = f"ratelimit:{user_id}"
    now = monotonic()

    allowed = script(keys=[key], args=[capacity, refill_rate, now])

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for '{user_id}'. "
            "Wait a moment before your next request."
        )


rate_limiter = TokenBucketRateLimiter(10, 1.0)
