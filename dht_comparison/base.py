"""Base classes and utilities for DHT protocol comparison.

Provides the abstract base class for DHT nodes, an in-process network
simulator, and helper functions for generating deterministic node IDs
and test keys.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LookupResult:
    """Result of a DHT key lookup."""
    key: int
    responsible_node: int
    hop_count: int
    path: list
    success: bool


class NetworkSimulator:
    """In-process network simulator for DHT nodes.

    All nodes live in the same Python process and communicate via direct
    method calls.  The simulator acts as a registry so any node can
    reference any other node by ID.
    """

    def __init__(self):
        self.nodes: dict[int, Any] = {}

    def register(self, node: "DHTNode"):
        self.nodes[node.node_id] = node

    def unregister(self, node_id: int):
        self.nodes.pop(node_id, None)

    def get_node(self, node_id: int) -> Optional["DHTNode"]:
        return self.nodes.get(node_id)

    @property
    def node_count(self) -> int:
        return len(self.nodes)


class DHTNode(ABC):
    """Abstract base class that every DHT protocol must implement."""

    def __init__(self, node_id: int, network: NetworkSimulator,
                 id_bits: int = 16):
        self.node_id = node_id
        self.network = network
        self.id_bits = id_bits
        self.id_space = 2 ** id_bits
        self.data: dict[int, Any] = {}

    @abstractmethod
    def join(self, bootstrap_id: Optional[int] = None) -> int:
        """Join the DHT.  Returns approximate message count."""
        ...

    @abstractmethod
    def leave(self) -> int:
        """Leave the DHT gracefully.  Returns approximate message count."""
        ...

    @abstractmethod
    def lookup(self, key: int) -> LookupResult:
        """Find the node responsible for *key*."""
        ...

    @abstractmethod
    def store(self, key: int, value: Any) -> bool:
        """Store a key-value pair in the DHT."""
        ...

    @abstractmethod
    def retrieve(self, key: int) -> Optional[Any]:
        """Retrieve a value by key from the DHT."""
        ...

    @abstractmethod
    def stabilize(self):
        """Run one round of the periodic stabilization protocol."""
        ...

    @abstractmethod
    def routing_table_size(self) -> int:
        """Number of distinct entries in the routing table."""
        ...

    @staticmethod
    def hash_key(key: str, id_bits: int = 16) -> int:
        """Hash a string key into the DHT ID space."""
        h = hashlib.sha1(key.encode()).hexdigest()
        return int(h, 16) % (2 ** id_bits)


# ---------------------------------------------------------------------------
# Deterministic ID / key generators (for reproducible benchmarks)
# ---------------------------------------------------------------------------

def generate_node_ids(count: int, id_bits: int = 16,
                      seed: int = 42) -> list[int]:
    """Return *count* well-distributed, deterministic node IDs."""
    ids: set[int] = set()
    i = 0
    while len(ids) < count:
        h = hashlib.sha1(f"node-{seed}-{i}".encode()).hexdigest()
        ids.add(int(h, 16) % (2 ** id_bits))
        i += 1
    return sorted(ids)[:count]


def generate_keys(count: int, id_bits: int = 16,
                  seed: int = 123) -> list[int]:
    """Return *count* deterministic test keys."""
    keys: set[int] = set()
    i = 0
    while len(keys) < count:
        h = hashlib.sha1(f"key-{seed}-{i}".encode()).hexdigest()
        keys.add(int(h, 16) % (2 ** id_bits))
        i += 1
    return sorted(keys)[:count]
