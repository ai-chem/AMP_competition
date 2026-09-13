"""Download putative non-AMP sequences from UniProtKB."""

from __future__ import annotations

from pathlib import Path

import requests

from amp_competition.features.physchem_constants import UNIPROT_QUERY


def download_uniprot_fasta(path: Path, *, query: str = UNIPROT_QUERY) -> Path:
    """
    Download all UniProtKB records matching the UniAMP-inspired query.

    The stream endpoint avoids pagination and writes the response incrementally
    to disk instead of holding the complete FASTA in memory.
    """
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "format": "fasta",
        "compressed": "true",
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, params=params, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with open(path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    return path
