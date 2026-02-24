"""Kademlia DHT implementation.

Reference
---------
Maymounkov & Mazières, "Kademlia: A Peer-to-peer Information System
Based on the XOR Metric" (IPTPS 2002)

Key properties
--------------
- XOR-based distance metric  d(a,b) = a XOR b
- k-bucket routing tables (one bucket per bit of the ID space)
- Iterative parallel lookups with alpha concurrent queries
- Responsible node = closest by XOR distance
"""

from typing import Any, Optional

from .base import DHTNode, LookupResult, NetworkSimulator


class KademliaNode(DHTNode):

    def __init__(self, node_id: int, network: NetworkSimulator,
                 id_bits: int = 16, k: int = 8, alpha: int = 3):
        super().__init__(node_id, network, id_bits)
        self.k = k          # max entries per bucket
        self.alpha = alpha   # lookup parallelism
        self.buckets: list[list[int]] = [[] for _ in range(id_bits)]

    # ------------------------------------------------------------------
    # XOR helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(a: int, b: int) -> int:
        return a ^ b

    def _bucket_index(self, node_id: int) -> int:
        dist = self._distance(self.node_id, node_id)
        if dist == 0:
            return 0
        return dist.bit_length() - 1

    # ------------------------------------------------------------------
    # Bucket management
    # ------------------------------------------------------------------

    def _update_bucket(self, node_id: int):
        if node_id == self.node_id:
            return
        idx = self._bucket_index(node_id)
        bucket = self.buckets[idx]
        if node_id in bucket:
            bucket.remove(node_id)
            bucket.append(node_id)      # move to tail (most-recently seen)
        elif len(bucket) < self.k:
            bucket.append(node_id)
        # bucket full → real Kademlia pings head; we skip for simplicity

    def _find_closest_local(self, target: int,
                            count: Optional[int] = None) -> list[int]:
        """Return up to *count* closest known nodes to *target*."""
        if count is None:
            count = self.k
        all_nodes = [nid for bucket in self.buckets for nid in bucket]
        all_nodes.sort(key=lambda nid: self._distance(nid, target))
        return all_nodes[:count]

    # ------------------------------------------------------------------
    # RPC handlers (called by other nodes during lookup)
    # ------------------------------------------------------------------

    def find_node_rpc(self, target: int, querier_id: int) -> list[int]:
        """FIND_NODE: return k closest known nodes to *target*."""
        self._update_bucket(querier_id)
        closest = self._find_closest_local(target)
        # Include self as a candidate (real implementations do this)
        candidates = closest + [self.node_id]
        candidates.sort(key=lambda nid: self._distance(nid, target))
        return candidates[:self.k]

    # ------------------------------------------------------------------
    # Iterative lookup
    # ------------------------------------------------------------------

    def lookup(self, key: int) -> LookupResult:
        shortlist = self._find_closest_local(key, self.k)
        if not shortlist:
            return LookupResult(key, self.node_id, 0, [self.node_id], True)

        queried: set[int] = {self.node_id}
        path = [self.node_id]
        hops = 0

        while True:
            to_query = [n for n in shortlist if n not in queried][:self.alpha]
            if not to_query:
                break

            hops += 1
            if getattr(self.network, "per_hop_delay", 0) > 0:
                advance = getattr(self.network, "advance_time", None)
                if advance is not None:
                    advance()
            found_new = False

            for nid in to_query:
                queried.add(nid)
                path.append(nid)

                target_node = self.network.get_node(nid)
                if target_node is None:
                    continue

                returned = target_node.find_node_rpc(key, self.node_id)
                for r in returned:
                    self._update_bucket(r)
                    if r not in shortlist and r != self.node_id:
                        shortlist.append(r)
                        found_new = True

            shortlist.sort(key=lambda n: self._distance(n, key))
            shortlist = shortlist[:self.k]

            if not found_new:
                break
            if hops > self.id_bits * 2:
                break

        # The responsible node is the closest known node (including self)
        all_candidates = shortlist + [self.node_id]
        responsible = min(all_candidates,
                          key=lambda n: self._distance(n, key))
        return LookupResult(key, responsible, hops, path, True)

    # ------------------------------------------------------------------
    # Join / Leave
    # ------------------------------------------------------------------

    def join(self, bootstrap_id: Optional[int] = None) -> int:
        self.network.register(self)
        if bootstrap_id is None:
            return 0

        self._update_bucket(bootstrap_id)

        # Self-lookup populates our buckets via iterative discovery
        result = self.lookup(self.node_id)
        return result.hop_count

    def leave(self) -> int:
        self.network.unregister(self.node_id)
        return 0

    # ------------------------------------------------------------------
    # Stabilization
    # ------------------------------------------------------------------

    def stabilize(self):
        # Remove dead nodes from every bucket
        for bucket in self.buckets:
            bucket[:] = [nid for nid in bucket
                         if self.network.get_node(nid) is not None]

    # ------------------------------------------------------------------
    # Store / Retrieve
    # ------------------------------------------------------------------

    def store(self, key: int, value: Any) -> bool:
        result = self.lookup(key)
        if not result.success:
            return False
        target = self.network.get_node(result.responsible_node)
        if target is None:
            return False
        target.data[key] = value
        return True

    def retrieve(self, key: int) -> Optional[Any]:
        result = self.lookup(key)
        if not result.success:
            return None
        target = self.network.get_node(result.responsible_node)
        if target is None:
            return None
        return target.data.get(key)

    def routing_table_size(self) -> int:
        return sum(len(b) for b in self.buckets)

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    @staticmethod
    def ground_truth(key: int, node_ids: list[int]) -> int:
        """Correct responsible node: closest by XOR distance."""
        return min(node_ids, key=lambda nid: nid ^ key)
