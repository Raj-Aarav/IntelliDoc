# cache_utils.py

from cache import AsyncLRU

@AsyncLRU(maxsize=256)
async def get_cached(func, *args, **kwargs):
    return await func(*args, **kwargs)
