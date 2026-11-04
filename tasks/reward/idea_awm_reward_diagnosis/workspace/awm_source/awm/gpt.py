import os
import asyncio
import concurrent.futures
import time
import json
from openai import AsyncAzureOpenAI, BadRequestError, AsyncOpenAI, InternalServerError
from openai.types.chat import ChatCompletion
from loguru import logger
from tqdm.asyncio import tqdm


class ChatCompletionFallback(dict):
    def __init__(self, data: dict[str, any]):
        super().__init__()
        for k, v in data.items():
            self[k] = self._wrap(v)

    def __getattr__(self, name: str) -> any:
        if name in self:
            return self[name]
        if name == "message" and "choices" in self and isinstance(self["choices"], list):
            first_choice = self["choices"][0]
            return first_choice.message
        return super().__getattribute__(name)

    @classmethod
    def _wrap(cls, obj: any) -> any:
        if isinstance(obj, dict):
            return cls(obj)
        if isinstance(obj, list):
            return [cls._wrap(o) for o in obj]
        return obj


class GPTClient:
    def __init__(self, provider: str = "azure", api_key: str | None = None, base_url: str | None = None, timeout: float = 600.0, max_retry_num: int = 3, retry_delay_seconds: float = 3, concurrency_limit: int = 64):

        if os.environ.get('AWM_SYN_LLM_PROVIDER'):
            provider = os.environ.get('AWM_SYN_LLM_PROVIDER')
            logger.info(f"Using provider from environment variable: {provider} overriding the args provider: {provider}")

        if os.environ.get('AWM_SYN_OVERRIDE_MODEL'):
            self._override_model = os.environ.get('AWM_SYN_OVERRIDE_MODEL')
            logger.warning(f"Using model from environment variable: {self._override_model} overriding any requested model")
        else:
            self._override_model = None

        assert provider in ['azure', 'openai'], "Invalid provider, must be 'azure' or 'openai'. For openai, you can set custom base url to adapt to your own inference service."

        if provider == 'azure':
            self.endpoint = base_url or os.getenv("AZURE_ENDPOINT_URL")
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        elif provider == 'openai':
            self.endpoint = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError(f"Invalid provider: {provider}")

        assert self.endpoint is not None and self.api_key is not None, "Endpoint and API key are required"
        
        self.timeout = timeout

        if provider == "azure":
            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version='2025-01-01-preview',
                timeout=self.timeout,
            )
        elif provider == "openai":
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.endpoint,
                timeout=self.timeout,
            )

        self._max_retry_num = max_retry_num
        self._retry_delay_seconds = retry_delay_seconds
        self._concurrency_limit = concurrency_limit
        self._semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}
        self.log_once = True
        self.provider = provider

    def _obj_to_plain(self, obj: any) -> any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._obj_to_plain(o) for o in obj]
        if isinstance(obj, dict):
            return {k: self._obj_to_plain(v) for k, v in obj.items()}
        if hasattr(obj, "to_dict"):
            return self._obj_to_plain(obj.to_dict())
        if hasattr(obj, "dict"):
            return self._obj_to_plain(obj.dict())
        if hasattr(obj, "__dict__"):
            return {k: self._obj_to_plain(v) for k, v in vars(obj).items()}
        return obj

    def _wrap_response(self, raw: any) -> ChatCompletionFallback:
        plain = self._obj_to_plain(raw)
        return ChatCompletionFallback(plain)

    def _build_refusal_completion(self, model: str, cf_results: dict[str, any]) -> ChatCompletionFallback:
        fallback = {
            "id": None,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": "",
                        "refusal": True,
                        "role": "assistant",
                        "annotations": [],
                        "audio": None,
                        "function_call": None,
                        "tool_calls": None,
                    },
                }
            ],
            "usage": None,
            "prompt_filter_results": [
                {"prompt_index": 0, "content_filter_results": cf_results}
            ],
        }
        return ChatCompletionFallback(fallback)

    def _get_semaphore(self) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.Semaphore(1)
        if loop not in self._semaphores:
            self._semaphores[loop] = asyncio.Semaphore(self._concurrency_limit)
        return self._semaphores[loop]

    async def _call_async(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None,
        max_completion_tokens: int | None,
        semaphore: asyncio.Semaphore | None = None,
        **kwargs
    ) -> ChatCompletionFallback:
        params = {"model": model, "messages": messages}
        if self._override_model:
            params["model"] = self._override_model
        
        if model.startswith('gpt-5'):
            temperature = 1.0

        params.update(kwargs)
        if temperature is not None:
            params["temperature"] = temperature
        if max_completion_tokens is not None:
            params["max_completion_tokens"] = max_completion_tokens
        if 'max_tokens' in params:
            params['max_completion_tokens'] = params['max_tokens']
            del params['max_tokens']
        last_error = None
        retry_delay = self._retry_delay_seconds
        sem = semaphore or self._get_semaphore()
        for attempt in range(self._max_retry_num):
            try:
                async with sem:
                    raw: ChatCompletion = await self._client.chat.completions.create(**params)
                if attempt > 0:
                    logger.info(f"Retry {attempt + 1}/{self._max_retry_num} succeeded")
                if self.log_once:
                    self.log_once = False
                    logger.info(f"GPT client request with params:\n{json.dumps(params, indent=4)}\nmodel={model}\nresponse={raw}")
                    logger.info(f"Using model: {params['model']} for request")
                return self._wrap_response(raw)
            except (BadRequestError, InternalServerError) as e:
                cf_results = {}
                code = None
                try:
                    err = e.response.json().get("error", {})
                    code = err.get("code") or err.get("innererror", {}).get("code")
                    cf_results = err.get("innererror", {}).get("content_filter_result", {}) or {}
                except Exception:
                    pass
                logger.error(f"Error in GPT request: {e}, code: {code}, content_filter_result: {cf_results}, return refusal completion. Input=\n{json.dumps(params, indent=4)}")
                last_error = e
                if attempt < self._max_retry_num - 1:
                    logger.warning(f"BadRequestError, InternalServerError on attempt {attempt + 1}/{self._max_retry_num}: {e}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
            except Exception as e:
                logger.error(f"Error in GPT request: {e}")
                last_error = e
                if attempt < self._max_retry_num - 1:
                    logger.warning(f"Connection error on attempt {attempt + 1}/{self._max_retry_num}: {e}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    break
        logger.error(f"Max retries exceeded. Last error: {last_error}")
        return self._build_refusal_completion(model, {"error": f"Max retries exceeded: {str(last_error)}"})

    async def send_request(
        self,
        messages: list[dict[str, str]],
        model: str = "your-llm-model-name",
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        **kwargs
    ) -> ChatCompletionFallback:

        return await self._call_async(messages, model, temperature, max_completion_tokens, semaphore=semaphore, **kwargs)

    def request(
        self,
        messages: list[dict[str, str]],
        model: str = "your-llm-model-name",
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        **kwargs
    ) -> ChatCompletionFallback:
        return self._run_async(self._request_async(messages, model, temperature, max_completion_tokens, **kwargs))

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str = "your-llm-model-name",
        stream: bool = False,
        **kwargs
    ) -> str:
        response = self.request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **kwargs
        )
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as e:
            logger.error(f"Failed to extract content from response: {e}")
            logger.error(f"Response structure: {response}")
            return ""

    def batch_requests(
        self,
        requests: list[dict[str, any]],
        progress_bar: bool = True,
    ) -> list[ChatCompletionFallback]:
        return self._run_async(self._batch_requests_async(requests, progress_bar))

    def batch_chat_completion(
        self,
        requests: list[dict[str, any]],
        progress_bar: bool = True,
    ) -> list[str]:
        responses = self.batch_requests(requests, progress_bar)
        contents = []
        for i, response in enumerate(responses):
            try:
                content = response.choices[0].message.content
                contents.append(content)
            except (AttributeError, IndexError, KeyError) as e:
                logger.error(f"Failed to extract content from response {i}: {e}")
                logger.error(f"Response structure: {response}")
                contents.append("")
        return contents

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    async def _request_async(
        self,
        messages: list[dict[str, str]],
        model: str = "your-llm-model-name",
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        **kwargs
    ) -> ChatCompletionFallback:
        return await self.send_request(messages, model, temperature, max_completion_tokens, **kwargs)

    async def _batch_requests_async(
        self,
        requests: list[dict[str, any]],
        progress_bar: bool = True,
    ) -> list[ChatCompletionFallback]:
        tasks = []
        sem = self._get_semaphore()
        for req in requests:
            task = self.send_request(
                req["messages"],
                req.get("model", "your-llm-model-name"),
                req.get("temperature"),
                req.get("max_completion_tokens"),
                semaphore=sem,
                **{k: v for k, v in req.items() if k not in ("messages", "model", "temperature", "max_completion_tokens")},
            )
            tasks.append(task)
        if progress_bar:
            return await tqdm.gather(*tasks, desc="Processing requests")
        else:
            return await asyncio.gather(*tasks)

    async def request_async(
        self,
        messages: list[dict[str, str]],
        model: str = "your-llm-model-name",
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        **kwargs
    ) -> ChatCompletionFallback:
        return await self._request_async(messages, model, temperature, max_completion_tokens, **kwargs)

    async def chat_completion_async(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str = "your-llm-model-name",
        stream: bool = False,
        **kwargs
    ) -> str:
        response = await self.request_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **kwargs
        )
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as e:
            logger.error(f"Failed to extract content from response: {e}")
            logger.error(f"Response structure: {response}")
            return ""

    async def batch_chat_completion_async(
        self,
        requests: list[dict[str, any]],
        progress_bar: bool = True,
    ) -> list[str]:
        responses = await self.batch_requests_async(requests, progress_bar)
        contents = []
        for i, response in enumerate(responses):
            try:
                content = response.choices[0].message.content
                contents.append(content)
            except (AttributeError, IndexError, KeyError) as e:
                logger.error(f"Failed to extract content from response {i}: {e}")
                logger.error(f"Response structure: {response}")
                contents.append("")
        return contents

    async def batch_requests_async(
        self,
        requests: list[dict[str, any]],
        progress_bar: bool = True,
    ) -> list[ChatCompletionFallback]:
        return await self._batch_requests_async(requests, progress_bar)
