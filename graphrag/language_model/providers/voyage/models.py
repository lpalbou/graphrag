# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Voyage AI LLM models implementation."""

from __future__ import annotations

import asyncio
import logging
import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING

import voyageai

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.language_model.protocol.base import EmbeddingModel
from graphrag.language_model.providers.anthropic.utils import run_coroutine_sync

# Use TYPE_CHECKING for type hints without runtime imports
if TYPE_CHECKING:
    from graphrag.config.models.language_model_config import LanguageModelConfig

log = logging.getLogger(__name__)


class VoyageEmbeddingModel(EmbeddingModel):
    """Voyage AI Embedding Model implementation."""

    def __init__(
        self,
        *,
        name: str,
        config: "LanguageModelConfig",  # Use string type annotation
        callbacks: Optional[WorkflowCallbacks] = None,
        cache: Optional[PipelineCache] = None,
    ) -> None:
        """
        Initialize the Voyage AI Embedding Model.

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
        
        # Create Voyage client using VOYAGE_API_KEY environment variable
        # Note: using a separate API key for Voyage, not the Anthropic key
        self.api_key = config.api_key
        if self.api_key is None or len(self.api_key.strip()) == 0:
            msg = "Voyage AI API key is required"
            raise ValueError(msg)
            
        self.client = voyageai.Client(api_key=self.api_key)
        
        # Set model parameters - Voyage uses voyage-2 model by default
        self.model = config.model or "voyage-2"
        
        # Voyage embedding dimensions
        self.embedding_dim = 1024  # Default for voyage-2 model

    async def aembed_batch(self, text_list: List[str], **kwargs) -> List[List[float]]:
        """
        Async embedding batch method for Voyage AI.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embeddings as float lists
        """
        if not text_list:
            return []
            
        try:
            # Use the specified model or default to voyage-2
            model = kwargs.get("model", self.model)
            
            # Make the API call to batch embed texts
            response = await asyncio.to_thread(
                self.client.embed, 
                texts=text_list,
                model=model
            )
            
            # Extract embeddings from the response
            if "embeddings" in response and response["embeddings"]:
                embeddings = response["embeddings"]
                return embeddings
            else:
                log.warning("No embeddings found in Voyage AI response")
                # Return zero embeddings as fallback
                return [np.zeros(self.embedding_dim).tolist() for _ in range(len(text_list))]
                
        except Exception as e:
            if self.callbacks:
                self.callbacks.error(f"Error during Voyage AI embedding request: {str(e)}")
            log.error(f"Error generating embeddings with Voyage AI: {str(e)}")
            
            # Return zero embeddings as fallback
            return [np.zeros(self.embedding_dim).tolist() for _ in range(len(text_list))]

    async def aembed(self, text: str, **kwargs) -> List[float]:
        """
        Async embedding method for Voyage AI.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding as a list of floats
        """
        if not text or text.strip() == "":
            return np.zeros(self.embedding_dim).tolist()
            
        embeddings = await self.aembed_batch([text], **kwargs)
        return embeddings[0] if embeddings else np.zeros(self.embedding_dim).tolist()

    def embed_batch(self, text_list: List[str], **kwargs) -> List[List[float]]:
        """
        Synchronous embedding batch method for Voyage AI.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embeddings as float lists
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Synchronous embedding method for Voyage AI.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding as a list of floats
        """
        return run_coroutine_sync(self.aembed(text, **kwargs)) 