#!/usr/bin/env python3
"""
Dataset loader that builds a PyTorch DataLoader from DatasetChain sources.

Supports URIs:
 - file://path/to/file
 - https://... (line-delimited)
 - ipfs://<CID>/<path> via HTTP gateway (env IPFS_GATEWAY, default https://ipfs.io/ipfs/)

Each line is treated as a training sample. If JSON, use fields 'input'/'target' if present,
else use the entire line as input. For causal LM, labels mirror input_ids.
"""

from __future__ import annotations

import io
import json
import urllib.request
from urllib.parse import urlparse
import os
from typing import Iterator, List, Dict, Any, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader


def _open_uri_lines(uri: str) -> Iterator[str]:
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()
    if scheme == 'file':
        path = parsed.path
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.rstrip('\n')
    elif scheme in ('http', 'https'):
        with urllib.request.urlopen(uri) as resp:  # nosec - controlled source by config
            for raw in resp:
                try:
                    line = raw.decode('utf-8').rstrip('\n')
                except Exception:
                    continue
                yield line
    elif scheme == 'ipfs':
        gateway = os.getenv('IPFS_GATEWAY', 'https://ipfs.io/ipfs')
        cid_and_path = parsed.netloc + parsed.path
        if gateway.endswith('/'):
            url = f"{gateway}{cid_and_path}"
        else:
            url = f"{gateway}/{cid_and_path}"
        yield from _open_uri_lines(url)
    else:
        raise ValueError(f"Unsupported URI scheme: {scheme}")


class URILineIterableDataset(IterableDataset):
    def __init__(self, uris: List[str]):
        super().__init__()
        self.uris = uris

    def __iter__(self):
        for uri in self.uris:
            for line in _open_uri_lines(uri):
                yield line


def _extract_text(line: str) -> str:
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            if 'input' in obj:
                return str(obj['input'])
            if 'text' in obj:
                return str(obj['text'])
    except Exception:
        pass
    return line


def _collate_tokenize(batch_lines: List[str], tokenizer, seq_len: int) -> Dict[str, torch.Tensor]:
    texts = [_extract_text(x) for x in batch_lines]
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=seq_len,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids']
    labels = input_ids.clone()
    out = {
        'input_ids': input_ids,
        'attention_mask': encoded.get('attention_mask', torch.ones_like(input_ids)),
        'labels': labels,
    }
    return out


def build_training_dataloader(
    dataset_chain,
    tokenizer,
    batch_size: int,
    seq_len: int,
    max_uris: Optional[int] = None,
) -> DataLoader:
    """Construct a DataLoader from GOLD-tier dataset URIs using the provided tokenizer."""
    from backend.core.dataset_block import DatasetQualityTier
    uris: List[str] = []
    try:
        gold_ids = dataset_chain.get_datasets_by_tier(DatasetQualityTier.GOLD)
        for dsid in gold_ids:
            info = dataset_chain.get_dataset_info(dsid)
            if not info:
                continue
            md = info.get('metadata', {})
            uri = md.get('source_uri')
            if uri:
                uris.append(uri)
            if max_uris and len(uris) >= max_uris:
                break
    except Exception:
        pass
    if not uris:
        raise RuntimeError("No dataset URIs available in GOLD tier")

    dataset = URILineIterableDataset(uris)
    collate = lambda lines: _collate_tokenize(lines, tokenizer, seq_len)  # noqa: E731
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    return loader

