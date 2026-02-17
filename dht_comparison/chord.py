"""Chord DHT implementation.

Reference
---------
Stoica et al., "Chord: A Scalable Peer-to-peer Lookup Service
for Internet Applications" (SIGCOMM 2001)

Key properties
--------------
- Nodes arranged on an identifier ring of size 2^m
- Each node maintains a finger table with O(log N) entries
- Lookups resolved in O(log N) hops
- Responsible node = successor of the key on the ring
"""

from typing import Any, Optional

from .base import DHTNode, LookupResult, NetworkSimulator


class ChordNode(DHTNode):

    def __init__(self, node_id: int, network: NetworkSimulator,
                 id_bits: int = 16):
        super().__init__(node_id, network, id_bits)
        self.successor: int = self.node_id
        self.predecessor: Optional[int] = None
        self.finger_table: list[int] = [self.node_id] * id_bits
        self.successor_list: list[int] = []
        self.successor_list_size: int = 3

    # ------------------------------------------------------------------
    # Ring-interval helpers
    # ------------------------------------------------------------------

    def _in_open(self, x: int, a: int, b: int) -> bool:
        """Is *x* in the open interval (a, b) on the ring?"""
        if a == b:
            return x != a
        if a < b:
            return a < x < b
        return x > a or x < b

    def _in_half_open(self, x: int, a: int, b: int) -> bool:
        """Is *x* in the half-open interval (a, b] on the ring?"""
        if a == b:
            return True  # entire ring
        if a < b:
            return a < x <= b
        return x > a or x <= b

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def _closest_preceding_finger(self, key: int) -> int:
        """Return the finger closest to (but preceding) *key*."""
        for i in range(self.id_bits - 1, -1, -1):
            f = self.finger_table[i]
            if f != self.node_id and self._in_open(f, self.node_id, key):
                if self.network.get_node(f) is not None:
                    return f
        return self.node_id

    def lookup(self, key: int) -> LookupResult:
        current = self
        hops = 0
        path = [self.node_id]

        while hops <= self.id_bits * 2:
            if current._in_half_open(key, current.node_id, current.successor):
                return LookupResult(key, current.successor, hops, path, True)

            next_id = current._closest_preceding_finger(key)
            if next_id == current.node_id:
                return LookupResult(key, current.successor, hops, path, True)

            next_node = self.network.get_node(next_id)
            if next_node is None:
                return LookupResult(key, current.successor, hops, path, True)

            hops += 1
            path.append(next_id)
            current = next_node

        return LookupResult(key, -1, hops, path, False)

    # ------------------------------------------------------------------
    # Join / Leave
    # ------------------------------------------------------------------

    def join(self, bootstrap_id: Optional[int] = None) -> int:
        if bootstrap_id is None:
            self.predecessor = None
            self.successor = self.node_id
            self.finger_table = [self.node_id] * self.id_bits
            self.network.register(self)
            return 0

        bootstrap = self.network.get_node(bootstrap_id)

        # Find our successor via the existing ring
        result = bootstrap.lookup(self.node_id)
        messages = result.hop_count + 1
        self.successor = result.responsible_node
        self.finger_table[0] = self.successor
        self.predecessor = None

        # Build initial finger table
        for i in range(1, self.id_bits):
            target = (self.node_id + (1 << i)) % self.id_space
            result = bootstrap.lookup(target)
            self.finger_table[i] = result.responsible_node
            messages += result.hop_count + 1

        self.network.register(self)
        return messages

    def leave(self) -> int:
        messages = 0

        if self.successor != self.node_id:
            succ = self.network.get_node(self.successor)
            if succ:
                succ.predecessor = self.predecessor
                messages += 1

        if self.predecessor is not None and self.predecessor != self.node_id:
            pred = self.network.get_node(self.predecessor)
            if pred:
                pred.successor = self.successor
                pred.finger_table[0] = self.successor
                messages += 1

        # Transfer local data to successor
        if self.successor != self.node_id:
            succ = self.network.get_node(self.successor)
            if succ:
                succ.data.update(self.data)

        self.network.unregister(self.node_id)
        return messages

    # ------------------------------------------------------------------
    # Stabilization
    # ------------------------------------------------------------------

    def _notify(self, candidate_id: int):
        if candidate_id == self.node_id:
            return
        if self.predecessor is None or \
                self._in_open(candidate_id, self.predecessor, self.node_id):
            self.predecessor = candidate_id

    def stabilize(self):
        # --- fix successor / predecessor ---
        succ = self.network.get_node(self.successor)
        if succ is None:
            for backup in self.successor_list:
                if self.network.get_node(backup) is not None:
                    self.successor = backup
                    self.finger_table[0] = backup
                    succ = self.network.get_node(backup)
                    break
            if succ is None:
                self.successor = self.node_id
                return

        x = succ.predecessor
        if x is not None and x != self.node_id \
                and self.network.get_node(x) is not None \
                and self._in_open(x, self.node_id, self.successor):
            self.successor = x
            self.finger_table[0] = x

        succ_node = self.network.get_node(self.successor)
        if succ_node:
            succ_node._notify(self.node_id)

        # --- fix finger table ---
        for i in range(self.id_bits):
            target = (self.node_id + (1 << i)) % self.id_space
            result = self.lookup(target)
            if result.success:
                self.finger_table[i] = result.responsible_node

        # --- maintain successor list ---
        self.successor_list = []
        cur = self.successor
        for _ in range(self.successor_list_size):
            if cur == self.node_id:
                break
            self.successor_list.append(cur)
            node = self.network.get_node(cur)
            if node is None:
                break
            cur = node.successor

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
        return len(set(self.finger_table) - {self.node_id})

    # ------------------------------------------------------------------
    # Ground truth (for test correctness checks)
    # ------------------------------------------------------------------

    @staticmethod
    def ground_truth(key: int, node_ids: list[int]) -> int:
        """Correct responsible node: first node with ID >= key (wrapping)."""
        sorted_ids = sorted(node_ids)
        for nid in sorted_ids:
            if nid >= key:
                return nid
        return sorted_ids[0]
