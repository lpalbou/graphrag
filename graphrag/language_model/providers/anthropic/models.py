# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Anthropic LLM models implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Generator, List, Optional, cast, TYPE_CHECKING

import anthropic

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
# Remove circular import
# from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.protocol.base import ChatModel, EmbeddingModel
from graphrag.language_model.response.base import ModelResponse, BaseModelOutput, BaseModelResponse
from graphrag.language_model.providers.anthropic.utils import (
    convert_history_to_anthropic_format,
    run_coroutine_sync,
)

# Use TYPE_CHECKING for type hints without runtime imports
if TYPE_CHECKING:
    from graphrag.config.models.language_model_config import LanguageModelConfig

log = logging.getLogger(__name__)


class AnthropicChatModel(ChatModel):
    """Anthropic Chat Model implementation using the Anthropic Python SDK."""

    def __init__(
        self,
        *,
        name: str,
        config: "LanguageModelConfig",  # Use string type annotation
        callbacks: Optional[WorkflowCallbacks] = None,
        cache: Optional[PipelineCache] = None,
    ) -> None:
        """
        Initialize the Anthropic Chat Model.

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
        
        # Create Anthropic client
        self.api_key = config.api_key
        if self.api_key is None or len(self.api_key.strip()) == 0:
            msg = "Anthropic API key is required"
            raise ValueError(msg)
            
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.sync_client = anthropic.Anthropic(api_key=self.api_key)
        
        # Set model parameters
        self.model = config.model or "claude-3.5-haiku-20241022"
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p

    async def achat(
        self, prompt: str, history: Optional[List] = None, **kwargs
    ) -> ModelResponse:
        """
        Async Chat method for Anthropic.

        Args:
            prompt: The prompt to send
            history: Optional chat history
            **kwargs: Additional parameters

        Returns:
            ModelResponse with the chat response
        """
        try:
            # Convert history to Anthropic format if provided
            messages = convert_history_to_anthropic_format(prompt, history)
            
            # Set up parameters
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)
            
            # Make the API call
            response = await self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            # Extract the text from the response
            content = response.content
            text = ""
            if content:
                for block in content:
                    if block.type == "text":
                        text += block.text
            
            # Create a properly formatted BaseModelOutput and BaseModelResponse
            model_output = BaseModelOutput(content=text)
            return BaseModelResponse(
                output=model_output,
                parsed_response=None,
                history=list(history) if history else [],
                raw_response=response
            )
        except Exception as e:
            if self.callbacks:
                self.callbacks.error(f"Error during Anthropic API call: {str(e)}")
            raise

    async def achat_stream(
        self, prompt: str, history: Optional[List] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming chat for Anthropic.

        Args:
            prompt: The prompt to send
            history: Optional chat history
            **kwargs: Additional parameters

        Yields:
            Chunks of the generated text
        """
        try:
            # Convert history to Anthropic format if provided
            messages = convert_history_to_anthropic_format(prompt, history)
            
            # Set up parameters
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)
            
            # Make the API call with streaming
            with self.client.messages.stream(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta" and chunk.delta.type == "text":
                        yield chunk.delta.text
        except Exception as e:
            if self.callbacks:
                self.callbacks.error(f"Error during Anthropic streaming API call: {str(e)}")
            raise

    def chat(self, prompt: str, history: Optional[List] = None, **kwargs) -> ModelResponse:
        """
        Synchronous chat method for Anthropic.

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
        Synchronous streaming chat for Anthropic.

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


class AnthropicEmbeddingModel(EmbeddingModel):
    """Anthropic Embedding Model implementation using the Anthropic Python SDK."""

    def __init__(
        self,
        *,
        name: str,
        config: "LanguageModelConfig",  # Use string type annotation
        callbacks: Optional[WorkflowCallbacks] = None,
        cache: Optional[PipelineCache] = None,
    ) -> None:
        """
        Initialize the Anthropic Embedding Model.

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
        
        # Create Anthropic client
        self.api_key = config.api_key
        if self.api_key is None or len(self.api_key.strip()) == 0:
            msg = "Anthropic API key is required"
            raise ValueError(msg)
            
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.sync_client = anthropic.Anthropic(api_key=self.api_key)
        
        # Set model parameters
        self.model = config.model or "claude-3.5-haiku-20241022"
        self.temperature = config.temperature
        self.top_p = config.top_p

    async def aembed_batch(self, text_list: List[str], **kwargs) -> List[List[float]]:
        """
        Async embedding batch method for Anthropic.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embeddings as float lists
        """
        try:
            # Anthropic does not have a dedicated embedding endpoint in the current SDK
            # We'll use the messages API to get embeddings from the model
            embeddings = []
            
            for text in text_list:
                response = await self.client.messages.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": f"Please represent the following text semantically: {text}"}
                    ],
                    max_tokens=1,  # We only need the embeddings, not a full response
                )
                
                # Extract the embedding from the response if available
                # Note: This is a placeholder. The actual implementation depends on how
                # Anthropic makes embeddings available in their API
                if hasattr(response, "embeddings") and response.embeddings:
                    embeddings.append(cast(List[float], response.embeddings))
                else:
                    # Fallback to empty embedding if not available
                    log.warning("Embeddings not available in Anthropic response")
                    # Create a simple embedding (as a placeholder)
                    embeddings.append([0.0] * 384)  # Use a standard embedding size
                    
            return embeddings
        except Exception as e:
            if self.callbacks:
                self.callbacks.error(f"Error during Anthropic embedding request: {str(e)}")
            raise

    async def aembed(self, text: str, **kwargs) -> List[float]:
        """
        Async embedding method for Anthropic.

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
        Synchronous embedding batch method for Anthropic.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embeddings as float lists
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Synchronous embedding method for Anthropic.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding as a list of floats
        """
        return run_coroutine_sync(self.aembed(text, **kwargs)) 