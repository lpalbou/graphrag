# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""SentenceTransformer models implementation."""

from __future__ import annotations

import asyncio
import logging
import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.language_model.protocol.base import EmbeddingModel
from graphrag.language_model.providers.anthropic.utils import run_coroutine_sync

# Use TYPE_CHECKING for type hints without runtime imports
if TYPE_CHECKING:
    from graphrag.config.models.language_model_config import LanguageModelConfig

log = logging.getLogger(__name__)


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """SentenceTransformer Embedding Model implementation."""

    def __init__(
        self,
        *,
        name: str,
        config: "LanguageModelConfig",  # Use string type annotation
        callbacks: Optional[WorkflowCallbacks] = None,
        cache: Optional[PipelineCache] = None,
    ) -> None:
        """
        Initialize the SentenceTransformer Embedding Model.

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
        
        # Set model name - default to all-MiniLM-L6-v2 if not specified
        self.model_name = config.model or "all-MiniLM-L6-v2"
        
        # Load the SentenceTransformer model (lazy loading on first use)
        self._model = None
        
        # Get embedding dimensions from the model config or use default
        # all-MiniLM-L6-v2 has 384 dimensions
        self.embedding_dim = 384  
        
        log.info(f"Initialized SentenceTransformer with model: {self.model_name}")

    @property
    def model(self):
        """Lazy load the model when first needed."""
        if self._model is None:
            try:
                log.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                # Update embedding dimension based on the actual model
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
                log.info(f"Model loaded with embedding dimension: {self.embedding_dim}")
            except Exception as e:
                log.error(f"Failed to load SentenceTransformer model: {str(e)}")
                raise
        return self._model

    async def aembed_batch(self, text_list: List[str], **kwargs) -> List[List[float]]:
        """
        Async embedding batch method for SentenceTransformer.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embeddings as float lists
        """
        if not text_list:
            return []
            
        try:
            # Process in batches to avoid memory issues with large inputs
            batch_size = kwargs.get("batch_size", 32)
            
            # Run the embedding in a thread to avoid blocking the event loop
            embeddings = await asyncio.to_thread(
                self.model.encode,
                text_list,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalize for cosine similarity
            )
            
            # Convert numpy arrays to lists
            return embeddings.tolist()
                
        except Exception as e:
            if self.callbacks:
                self.callbacks.error(f"Error during SentenceTransformer embedding request: {str(e)}")
            log.error(f"Error generating embeddings with SentenceTransformer: {str(e)}")
            
            # Return zero embeddings as fallback
            return [np.zeros(self.embedding_dim).tolist() for _ in range(len(text_list))]

    async def aembed(self, text: str, **kwargs) -> List[float]:
        """
        Async embedding method for SentenceTransformer.

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
        Synchronous embedding batch method for SentenceTransformer.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embeddings as float lists
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Synchronous embedding method for SentenceTransformer.

        Args:
            text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embedding as a list of floats
        """
        return run_coroutine_sync(self.aembed(text, **kwargs)) 