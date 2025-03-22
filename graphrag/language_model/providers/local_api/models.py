# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Local API LLM models implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

import aiohttp
import requests

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.protocol.base import ChatModel, EmbeddingModel
from graphrag.language_model.response.base import ModelResponse
from graphrag.language_model.providers.local_api.utils import run_coroutine_sync

log = logging.getLogger(__name__)


class LocalAPIChatModel(ChatModel):
    """Local API Chat Model that connects to a locally served LLM API endpoint."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: Optional[WorkflowCallbacks] = None,
        cache: Optional[PipelineCache] = None,
    ) -> None:
        """
        Initialize the Local API Chat Model.

        Args:
            name: The name of the model
            config: The model configuration
            callbacks: Optional callbacks
            cache: Optional cache
        """
        self.name = name
        self.config = config
        self.callbacks = callbacks
        self.cache = cache
        self.api_url = config.api_base or "http://127.0.0.1:8000/generate"
        self.request_timeout = config.request_timeout

    async def achat(
        self, prompt: str, history: Optional[List] = None, **kwargs
    ) -> ModelResponse:
        """
        Async Chat method for Local API.

        Args:
            prompt: The prompt to send
            history: Optional chat history
            **kwargs: Additional parameters

        Returns:
            ModelResponse with the chat response
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "prompt": prompt,
                    "history": history or [],
                    "model": self.config.model,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "top_p": kwargs.get("top_p", self.config.top_p),
                }
                
                async with session.post(
                    self.api_url, 
                    json=payload,
                    timeout=self.request_timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Error from Local API: {error_text}")
                    
                    result = await response.json()
                    
                    # Adapt the response to match the expected format
                    return ModelResponse(
                        text=result.get("text", ""),
                        raw_response=result
                    )
        except Exception as e:
            if self.callbacks:
                self.callbacks.error(f"Error during Local API chat request: {str(e)}")
            raise

    async def achat_stream(
        self, prompt: str, history: Optional[List] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming chat for Local API.

        Args:
            prompt: The prompt to send
            history: Optional chat history
            **kwargs: Additional parameters

        Yields:
            Chunks of the generated text
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "prompt": prompt,
                    "history": history or [],
                    "model": self.config.model,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "top_p": kwargs.get("top_p", self.config.top_p),
                    "stream": True,
                }
                
                async with session.post(
                    self.api_url, 
                    json=payload,
                    timeout=self.request_timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Error from Local API: {error_text}")
                    
                    # Assuming the server sends SSE or another streaming format
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8').strip())
                                if "text" in data:
                                    yield data["text"]
                            except json.JSONDecodeError:
                                # If it's not JSON, yield the raw text
                                yield line.decode('utf-8').strip()
        except Exception as e:
            if self.callbacks:
                self.callbacks.error(f"Error during Local API chat stream: {str(e)}")
            raise

    def chat(self, prompt: str, history: Optional[List] = None, **kwargs) -> ModelResponse:
        """
        Synchronous chat method for Local API.

        Args:
            prompt: The prompt to send
            history: Optional chat history
            **kwargs: Additional parameters

        Returns:
            ModelResponse with the chat response
        """
        return run_coroutine_sync(self.achat(prompt, history, **kwargs))

    def chat_stream(
        self, prompt: str, history: Optional[List] = None, **kwargs
    ) -> Generator[str, None, None]:
        """
        Synchronous streaming chat for Local API.

        Args:
            prompt: The prompt to send
            history: Optional chat history
            **kwargs: Additional parameters

        Yields:
            Chunks of the generated text
        """
        async def _run_stream():
            async for chunk in self.achat_stream(prompt, history, **kwargs):
                yield chunk
        
        return run_coroutine_sync(_run_stream().__aiter__().__anext__)


class LocalAPIEmbeddingModel(EmbeddingModel):
    """Local API Embedding Model that connects to a locally served embedding API endpoint."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: Optional[WorkflowCallbacks] = None,
        cache: Optional[PipelineCache] = None,
    ) -> None:
        """
        Initialize the Local API Embedding Model.

        Args:
            name: The name of the model
            config: The model configuration
            callbacks: Optional callbacks
            cache: Optional cache
        """
        self.name = name
        self.config = config
        self.callbacks = callbacks
        self.cache = cache
        # Use a different endpoint for embeddings
        self.api_url = config.api_base or "http://127.0.0.1:8000/embeddings"
        self.request_timeout = config.request_timeout

    async def aembed_batch(self, text_list: List[str], **kwargs) -> List[List[float]]:
        """
        Async embedding batch method for Local API.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embeddings as float lists
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "texts": text_list,
                    "model": self.config.model,
                }
                
                async with session.post(
                    self.api_url, 
                    json=payload,
                    timeout=self.request_timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Error from Local API: {error_text}")
                    
                    result = await response.json()
                    
                    # Expect a list of embeddings
                    return result.get("embeddings", [])
        except Exception as e:
            if self.callbacks:
                self.callbacks.error(f"Error during Local API embedding request: {str(e)}")
            raise

    async def aembed(self, text: str, **kwargs) -> List[float]:
        """
        Async embedding method for Local API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding as a list of floats
        """
        embeddings = await self.aembed_batch([text], **kwargs)
        return embeddings[0] if embeddings else []

    def embed_batch(self, text_list: List[str], **kwargs) -> List[List[float]]:
        """
        Synchronous embedding batch method for Local API.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embeddings as float lists
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Synchronous embedding method for Local API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding as a list of floats
        """
        return run_coroutine_sync(self.aembed(text, **kwargs)) 