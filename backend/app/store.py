from dataclasses import dataclass, field
from typing import Any


@dataclass
class InMemoryStore:
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    query_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_uploads: dict[str, dict[str, Any]] = field(default_factory=dict)


store = InMemoryStore()
