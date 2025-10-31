"""
MCP Server for Multilingual Data Augmentation
Provides text augmentation services via Model Context Protocol

Installation:
pip install mcp pandas numpy tqdm

Usage:
python augmentation_mcp_server.py
"""

import asyncio
import json
import logging
from typing import Any, Sequence, Optional, List, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import random
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
from mcp.server.stdio import stdio_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("augmentation-server")


# =============================================================================
# AUGMENTATION ENGINE (Core Logic)
# =============================================================================

class AugmentationEngine:
    """Core augmentation logic"""

    def __init__(self):
        self.cache = {}
        self.stats = {
            'total_requests': 0,
            'total_augmented': 0,
            'cache_hits': 0,
            'languages_processed': set()
        }

    @lru_cache(maxsize=5000)
    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _is_indic_script(self, text: str) -> bool:
        """Detect Indic scripts"""
        indic_ranges = [
            (0x0900, 0x097F),  # Devanagari
            (0x0980, 0x09FF),  # Bengali
            (0x0A00, 0x0A7F),  # Gurmukhi
            (0x0A80, 0x0AFF),  # Gujarati
            (0x0B00, 0x0B7F),  # Oriya
            (0x0B80, 0x0BFF),  # Tamil
            (0x0C00, 0x0C7F),  # Telugu
            (0x0C80, 0x0CFF),  # Kannada
            (0x0D00, 0x0D7F),  # Malayalam
        ]

        for char in text[:100]:
            code = ord(char)
            for start, end in indic_ranges:
                if start <= code <= end:
                    return True
        return False

    def random_deletion(self, text: str, p: float = 0.15) -> str:
        words = text.split()
        if len(words) <= 3:
            return text

        min_words = max(3, int(len(words) * 0.7))
        new_words = [w for w in words if random.random() > p]

        return ' '.join(new_words) if len(new_words) >= min_words else text

    def random_swap(self, text: str, n: Optional[int] = None) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        n = n or max(1, len(words) // 15)
        words_copy = words.copy()

        for _ in range(min(n, len(words) // 2)):
            idx1, idx2 = random.sample(range(len(words_copy)), 2)
            words_copy[idx1], words_copy[idx2] = words_copy[idx2], words_copy[idx1]

        return ' '.join(words_copy)

    def random_insertion(self, text: str, n: Optional[int] = None) -> str:
        words = text.split()
        if len(words) < 3:
            return text

        n = n or max(1, len(words) // 20)
        words_copy = words.copy()

        for _ in range(n):
            word_to_insert = random.choice(words)
            insert_pos = random.randint(0, len(words_copy))
            words_copy.insert(insert_pos, word_to_insert)

        return ' '.join(words_copy)

    def character_level_noise(self, text: str, p: float = 0.02) -> str:
        if not self._is_indic_script(text):
            return text

        chars = list(text)
        for i in range(len(chars)):
            if random.random() < p and chars[i] != ' ':
                operation = random.choice(['skip', 'duplicate'])
                if operation == 'skip' and i < len(chars) - 1:
                    chars[i] = ''
                elif operation == 'duplicate':
                    chars[i] = chars[i] * 2

        return ''.join(chars)

    def word_order_permutation(self, text: str, window_size: int = 3) -> str:
        words = text.split()
        if len(words) < window_size:
            return text

        result = []
        i = 0

        while i < len(words):
            window = words[i:i + window_size]
            if len(window) > 1 and random.random() > 0.5:
                random.shuffle(window)
            result.extend(window)
            i += window_size

        return ' '.join(result)

    def augment_text(self, text: str, n_augmentations: int = 2,
                     techniques: Optional[List[str]] = None) -> List[str]:
        """Generate augmentations for a single text"""
        self.stats['total_requests'] += 1

        # Check cache
        cache_key = self._get_text_hash(text)
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key][:n_augmentations]

        # Available techniques
        all_techniques = {
            'deletion': self.random_deletion,
            'swap': self.random_swap,
            'insertion': self.random_insertion,
            'char_noise': self.character_level_noise,
            'permutation': self.word_order_permutation
        }

        # Select techniques
        if techniques:
            selected = {k: v for k, v in all_techniques.items() if k in techniques}
        else:
            selected = all_techniques

        technique_list = list(selected.values())
        augmented = []
        attempts = 0
        max_attempts = n_augmentations * 3

        while len(augmented) < n_augmentations and attempts < max_attempts:
            selected_techs = random.sample(technique_list,
                                           random.randint(1, min(2, len(technique_list))))

            aug_text = text
            for technique in selected_techs:
                try:
                    aug_text = technique(aug_text)
                except Exception as e:
                    logger.warning(f"Augmentation failed: {e}")
                    continue

            if aug_text != text and aug_text not in augmented:
                augmented.append(aug_text)

            attempts += 1

        # Cache result
        self.cache[cache_key] = augmented
        self.stats['total_augmented'] += len(augmented)

        return augmented

    def augment_batch(self, texts: List[str], n_augmentations: int = 2,
                      use_parallel: bool = True) -> List[List[str]]:
        """Augment multiple texts efficiently"""

        if use_parallel and len(texts) > 10:
            n_workers = max(1, mp.cpu_count() - 1)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(
                    lambda t: self.augment_text(t, n_augmentations),
                    texts
                ))
            return results
        else:
            return [self.augment_text(t, n_augmentations) for t in texts]

    def get_stats(self) -> Dict:
        """Get server statistics"""
        return {
            'total_requests': self.stats['total_requests'],
            'total_augmented': self.stats['total_augmented'],
            'cache_hits': self.stats['cache_hits'],
            'cache_size': len(self.cache),
            'cache_hit_rate': f"{(self.stats['cache_hits'] / max(1, self.stats['total_requests']) * 100):.2f}%"
        }


# =============================================================================
# MCP SERVER IMPLEMENTATION
# =============================================================================

# Initialize augmentation engine
engine = AugmentationEngine()
app = Server("multilingual-augmentation-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available augmentation tools"""
    return [
        Tool(
            name="augment_text",
            description="""Augment a single text using multiple techniques.

Parameters:
- text: Text to augment (required)
- n_augmentations: Number of variations (default: 2)
- techniques: List of techniques to use (optional)
  Available: deletion, swap, insertion, char_noise, permutation

Returns: List of augmented texts""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to augment"
                    },
                    "n_augmentations": {
                        "type": "integer",
                        "description": "Number of augmented versions",
                        "default": 2
                    },
                    "techniques": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Augmentation techniques to use"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="augment_batch",
            description="""Augment multiple texts efficiently in batch.

Parameters:
- texts: List of texts to augment (required)
- n_augmentations: Number of variations per text (default: 2)
- use_parallel: Enable parallel processing (default: true)

Returns: List of augmented text lists""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Texts to augment"
                    },
                    "n_augmentations": {
                        "type": "integer",
                        "description": "Number of augmented versions per text",
                        "default": 2
                    },
                    "use_parallel": {
                        "type": "boolean",
                        "description": "Use parallel processing",
                        "default": True
                    }
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="augment_dataset",
            description="""Augment entire dataset from CSV format.

Parameters:
- csv_data: CSV string with columns (required)
- text_column: Name of text column (auto-detect if not provided)
- label_column: Name of label column (optional)
- n_augmentations: Augmentations per sample (default: 2)
- balance_labels: Balance minority classes (default: false)

Returns: Augmented CSV data""",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {
                        "type": "string",
                        "description": "CSV data as string"
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Text column name"
                    },
                    "label_column": {
                        "type": "string",
                        "description": "Label column name"
                    },
                    "n_augmentations": {
                        "type": "integer",
                        "default": 2
                    },
                    "balance_labels": {
                        "type": "boolean",
                        "default": False
                    }
                },
                "required": ["csv_data"]
            }
        ),
        Tool(
            name="get_server_stats",
            description="""Get server statistics and performance metrics.

Returns: Server statistics including cache performance and processing counts""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="clear_cache",
            description="""Clear the augmentation cache to free memory.

Returns: Confirmation message""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle tool calls"""

    try:
        if name == "augment_text":
            text = arguments.get("text")
            n_aug = arguments.get("n_augmentations", 2)
            techniques = arguments.get("techniques")

            if not text:
                return [TextContent(type="text", text="Error: 'text' parameter is required")]

            augmented = engine.augment_text(text, n_aug, techniques)

            result = {
                "original": text,
                "augmented": augmented,
                "count": len(augmented)
            }

            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "augment_batch":
            texts = arguments.get("texts", [])
            n_aug = arguments.get("n_augmentations", 2)
            use_parallel = arguments.get("use_parallel", True)

            if not texts:
                return [TextContent(type="text", text="Error: 'texts' parameter is required")]

            augmented_batch = engine.augment_batch(texts, n_aug, use_parallel)

            result = {
                "original_count": len(texts),
                "total_augmented": sum(len(aug) for aug in augmented_batch),
                "results": [
                    {"original": orig, "augmented": aug}
                    for orig, aug in zip(texts, augmented_batch)
                ]
            }

            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "augment_dataset":
            csv_data = arguments.get("csv_data")
            text_col = arguments.get("text_column")
            label_col = arguments.get("label_column")
            n_aug = arguments.get("n_augmentations", 2)
            balance = arguments.get("balance_labels", False)

            if not csv_data:
                return [TextContent(type="text", text="Error: 'csv_data' parameter is required")]

            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(csv_data))

            # Auto-detect text column if not provided
            if not text_col:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        avg_len = df[col].astype(str).str.len().mean()
                        if avg_len > 50:  # Likely text column
                            text_col = col
                            break

            if not text_col or text_col not in df.columns:
                return [TextContent(type="text", text="Error: Could not detect text column")]

            # Augment texts
            augmented_rows = []
            for _, row in df.iterrows():
                augmented_rows.append(row.to_dict())  # Original

                text = str(row[text_col])
                augmented_texts = engine.augment_text(text, n_aug)

                for aug_text in augmented_texts:
                    aug_row = row.to_dict()
                    aug_row[text_col] = aug_text
                    augmented_rows.append(aug_row)

            # Create augmented dataframe
            aug_df = pd.DataFrame(augmented_rows)

            # Convert to CSV
            output = StringIO()
            aug_df.to_csv(output, index=False)
            csv_result = output.getvalue()

            stats = {
                "original_samples": len(df),
                "augmented_samples": len(aug_df),
                "increase": len(aug_df) - len(df),
                "csv_data": csv_result
            }

            return [TextContent(type="text", text=json.dumps(stats, ensure_ascii=False, indent=2))]

        elif name == "get_server_stats":
            stats = engine.get_stats()
            return [TextContent(type="text", text=json.dumps(stats, indent=2))]

        elif name == "clear_cache":
            cache_size = len(engine.cache)
            engine.cache.clear()
            engine.stats['cache_hits'] = 0
            return [TextContent(type="text", text=f"Cache cleared. Removed {cache_size} entries.")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server"""
    logger.info("Starting Multilingual Augmentation MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())