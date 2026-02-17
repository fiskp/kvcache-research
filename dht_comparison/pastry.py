"""Pastry DHT implementation.

Reference
---------
Rowstron & Druschel, "Pastry: Scalable, Decentralized Object Location
and Routing for Large-Scale Peer-to-Peer Systems" (Middleware 2001)

Key properties
--------------
- Prefix-based routing using base-2^b digits
- Leaf set for local neighbourhood awareness
- O(log_{2^b} N) routing hops  (fewer than Chord/Kademlia for same N)
- Responsible node = numerically closest on the ID ring
"""

from typing import Any, Optional

from .base import DHTNode, LookupResult, NetworkSimulator


class PastryNode(DHTNode):

    def __init__(self, node_id: int, network: NetworkSimulator,
                 id_bits: int = 16, b: int = 4, leaf_size: int = 8):
        super().__init__(node_id, network, id_bits)
        self.b = b                      # bits per digit
        self.base = 1 << b              # digit values  (16 for b=4)
        self.num_digits = (id_bits + b - 1) // b
        self.leaf_size = leaf_size       # total leaf-set capacity

        # routing_table[row][col] — row = prefix length, col = digit value
        self.routing_table: list[list[Optional[int]]] = [
            [None] * self.base for _ in range(self.num_digits)
        ]
        # Leaf set: closest nodes by circular distance
        self.leaf_set: list[int] = []

    # ------------------------------------------------------------------
    # Digit / prefix helpers
    # ------------------------------------------------------------------

    def _get_digit(self, node_id: int, position: int) -> int:
        """Extract digit at *position* (0 = most-significant)."""
        shift = (self.num_digits - 1 - position) * self.b
        return (node_id >> shift) & (self.base - 1)

    def _shared_prefix_length(self, a: int, b: int) -> int:
        for i in range(self.num_digits):
            if self._get_digit(a, i) != self._get_digit(b, i):
                return i
        return self.num_digits

    def _circular_distance(self, a: int, b: int) -> int:
        diff = abs(a - b)
        return min(diff, self.id_space - diff)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _add_to_state(self, node_id: int):
        """Incorporate a discovered node into the leaf set and routing table."""
        if node_id == self.node_id:
            return
        if self.network.get_node(node_id) is None:
            return

        # --- leaf set ---
        if node_id not in self.leaf_set:
            self.leaf_set.append(node_id)
            self.leaf_set.sort(
                key=lambda x: self._circular_distance(self.node_id, x))
            if len(self.leaf_set) > self.leaf_size:
                self.leaf_set = self.leaf_set[:self.leaf_size]

        # --- routing table ---
        plen = self._shared_prefix_length(self.node_id, node_id)
        if plen < self.num_digits:
            digit = self._get_digit(node_id, plen)
            cur = self.routing_table[plen][digit]
            if cur is None or self.network.get_node(cur) is None:
                self.routing_table[plen][digit] = node_id

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_next(self, key: int) -> Optional[int]:
        """Return the next hop toward *key*, or ``None`` if we are closest."""
        my_dist = self._circular_distance(self.node_id, key)
        if my_dist == 0:
            return None

        # Step 1 — prefix-based routing table
        plen = self._shared_prefix_length(self.node_id, key)
        if plen < self.num_digits:
            digit = self._get_digit(key, plen)
            entry = self.routing_table[plen][digit]
            if entry is not None and self.network.get_node(entry) is not None:
                return entry

        # Step 2 — leaf set + full routing table: pick the closer node
        best: Optional[int] = None
        best_dist = my_dist

        for nid in self.leaf_set:
            if self.network.get_node(nid) is None:
                continue
            d = self._circular_distance(nid, key)
            if d < best_dist or (d == best_dist and best is not None
                                 and nid < best):
                best = nid
                best_dist = d

        for row in self.routing_table:
            for entry in row:
                if entry is None or self.network.get_node(entry) is None:
                    continue
                d = self._circular_distance(entry, key)
                if d < best_dist or (d == best_dist and best is not None
                                     and entry < best):
                    best = entry
                    best_dist = d

        return best

    def lookup(self, key: int) -> LookupResult:
        current = self
        hops = 0
        path = [self.node_id]
        visited: set[int] = {self.node_id}

        while hops <= self.id_bits * 2:
            next_hop = current._route_next(key)

            if next_hop is None or next_hop in visited:
                return LookupResult(key, current.node_id, hops, path, True)

            next_node = self.network.get_node(next_hop)
            if next_node is None:
                return LookupResult(key, current.node_id, hops, path, True)

            hops += 1
            path.append(next_hop)
            visited.add(next_hop)
            current = next_node

        return LookupResult(key, current.node_id, hops, path, False)

    # ------------------------------------------------------------------
    # Join / Leave
    # ------------------------------------------------------------------

    def join(self, bootstrap_id: Optional[int] = None) -> int:
        self.network.register(self)
        if bootstrap_id is None:
            return 0

        bootstrap = self.network.get_node(bootstrap_id)
        if bootstrap is None:
            return 0

        messages = 0

        # Copy bootstrap's full state
        self._add_to_state(bootstrap_id)
        for nid in bootstrap.leaf_set:
            self._add_to_state(nid)
        for row in bootstrap.routing_table:
            for entry in row:
                if entry is not None:
                    self._add_to_state(entry)
        messages += 1

        # Self-lookup to discover more nodes
        result = self.lookup(self.node_id)
        messages += result.hop_count

        # Learn state from every node on the path
        for nid in result.path:
            if nid == self.node_id:
                continue
            node = self.network.get_node(nid)
            if node is None or not isinstance(node, PastryNode):
                continue
            for leaf_id in node.leaf_set:
                self._add_to_state(leaf_id)
            for row in node.routing_table:
                for entry in row:
                    if entry is not None:
                        self._add_to_state(entry)
            messages += 1

        # Announce ourselves to leaf-set neighbours
        for leaf_id in self.leaf_set:
            leaf = self.network.get_node(leaf_id)
            if leaf is not None and isinstance(leaf, PastryNode):
                leaf._add_to_state(self.node_id)
                messages += 1

        return messages

    def leave(self) -> int:
        messages = 0

        # Notify leaf-set neighbours so they remove us
        for leaf_id in self.leaf_set:
            leaf = self.network.get_node(leaf_id)
            if leaf is None or not isinstance(leaf, PastryNode):
                continue
            if self.node_id in leaf.leaf_set:
                leaf.leaf_set.remove(self.node_id)
            for row in leaf.routing_table:
                for j in range(len(row)):
                    if row[j] == self.node_id:
                        row[j] = None
            messages += 1

        # Transfer data to closest surviving leaf
        if self.leaf_set:
            closest = min(
                self.leaf_set,
                key=lambda x: self._circular_distance(self.node_id, x))
            closest_node = self.network.get_node(closest)
            if closest_node is not None:
                closest_node.data.update(self.data)

        self.network.unregister(self.node_id)
        return messages

    # ------------------------------------------------------------------
    # Stabilization
    # ------------------------------------------------------------------

    def stabilize(self):
        # Remove dead entries
        self.leaf_set = [nid for nid in self.leaf_set
                         if self.network.get_node(nid) is not None]
        for i in range(self.num_digits):
            for j in range(self.base):
                entry = self.routing_table[i][j]
                if entry is not None \
                        and self.network.get_node(entry) is None:
                    self.routing_table[i][j] = None

        # Exchange state with leaf neighbours
        for leaf_id in list(self.leaf_set):
            leaf = self.network.get_node(leaf_id)
            if leaf is None or not isinstance(leaf, PastryNode):
                continue
            for other_id in leaf.leaf_set:
                self._add_to_state(other_id)
            for row in leaf.routing_table:
                for entry in row:
                    if entry is not None:
                        self._add_to_state(entry)

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
        count = len(self.leaf_set)
        for row in self.routing_table:
            count += sum(1 for e in row if e is not None)
        return count

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    @staticmethod
    def ground_truth(key: int, node_ids: list[int],
                     id_space: int = 65536) -> int:
        """Correct responsible node: numerically closest on the ring."""
        def circ(a: int, b: int) -> int:
            diff = abs(a - b)
            return min(diff, id_space - diff)
        return min(node_ids, key=lambda nid: (circ(nid, key), nid))
