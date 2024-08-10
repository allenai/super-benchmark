import hashlib
import os
import pickle
from dataclasses import dataclass

from dotenv import load_dotenv
from litellm import completion, completion_cost
from together import Together

load_dotenv()


def cache(avoid_fields=None, skip_load=False, cache_dir=".cache"):
    """Cache the output of a function call in a file under ".cache" directory."""

    def _hash(*args, **kwargs):
        """Hash the input arguments"""
        return hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()

    def _cache_file(_cache_id=None, *args, **kwargs):
        """Return the path to the cache file"""
        context_cache_dir = cache_dir
        filename = (_hash(*args, **kwargs) + "_" + str(_cache_id) if _cache_id else _hash(*args, **kwargs)) + ".pkl"
        return os.path.abspath(os.path.join(context_cache_dir, filename))

    def _cache(_cache_id=None, **kwargs):
        """Return the cached output if it exists, otherwise return None"""
        cache_file = _cache_file(_cache_id, **kwargs)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)["output"]
        else:
            return None

    def _save_cache(*args, **kwargs):
        """Save the output to the cache file"""
        output = kwargs.pop("output")
        cache_file = _cache_file(*args, **kwargs)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            dump = {"output": output, "input": {"args": args, "kwargs": kwargs}}
            pickle.dump(dump, f)

    def _decorator(func):
        def _wrapper(*args, **kwargs):
            # convert args and kwargs to a dictionary
            all_kwargs = dict(zip(func.__code__.co_varnames, args))
            all_kwargs.update(kwargs)
            all_kwargs = {k: v for k, v in all_kwargs.items() if k not in avoid_fields} if avoid_fields else all_kwargs

            cache_id = all_kwargs.pop("_cache_id", None)
            kwargs.pop("_cache_id", None)

            # Check if the output is cached
            try:
                cached = _cache(_cache_id=cache_id, **all_kwargs)
                if cached is not None and not skip_load:
                    return cached
            except Exception as e:
                pass

            # Call the function
            output = func(*args, **kwargs)

            # Save the output to the cache
            _save_cache(**all_kwargs, output=output)

            return output

        return _wrapper

    return _decorator

@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


@cache()
def generate_response(model_engine, prompt, stop_tokens=None, max_output_tokens=600, temperature=0.2, top_p=0.5) -> LLMResponse:
    together = "together_ai" in model_engine
    if together:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        model_engine = model_engine.replace("together_ai/", "")
        completion_func = client.chat.completions.create
    else:
        completion_func = completion

    response = completion_func(
        model=model_engine,
        messages=prompt,
        max_tokens=max_output_tokens,
        stop=stop_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    text = response.choices[0].message.content.strip()
    cost = completion_cost(completion_response=response, model=model_engine, messages=prompt) if not together else 0

    return LLMResponse(
        text=text,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        cost=cost,
    )
