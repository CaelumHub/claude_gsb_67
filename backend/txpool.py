"""In-memory transaction pool (mempool) with replace-by-fee support.

The pool holds signed, validated-but-unconfirmed transactions until a block is
mined.  It enforces:

* signature validity,
* sender authenticity (public key must match the sender address),
* nonce monotonicity (only the *next* nonce per sender is admitted, preventing
  nonce gaps and making double-spends structurally impossible),
* sufficient balance against the current world state,
* fee >= 0 and a bounded pool size.

Replace-by-fee (RBF)
--------------------
A sender may replace their pending transaction for a given nonce any number of
times, as long as every replacement:

* uses the same ``(sender, nonce)`` as the transaction currently in the pool,
* keeps the original type, recipient, amount and payload unchanged — only the
  fee (and therefore the txid) may change,
* offers a *strictly higher* fee than the transaction it replaces.

Transactions whose nonce is already confirmed on-chain can never be replaced
(the account nonce has moved past them), and a replacement never touches other
senders' pending transactions.

Every accepted replacement is recorded in an audit trail
(:attr:`TxPool._replacement_events`) so the superseded transactions remain
visible after they leave the pool.  The trail is persisted alongside the pool
and exposed through the API.

When a block is mined, included transactions are dropped (together with any
stale replaced variant of the same ``(sender, nonce)``); on a reorg, the
transactions from abandoned blocks are re-admitted so they are not lost.
"""

import copy
import time

from .config import TXPOOL_SORT_KEY
from .transaction import Transaction


class TxPool:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self._pool = {}               # txid -> Transaction
        self._order = []              # txids in arrival order
        self._by_sender = {}          # sender -> txid (one pending tx per sender)
        self._by_sender_nonce = {}    # (sender, nonce) -> txid
        # Replace-by-fee audit trail.
        self._replacement_events = []   # immutable, append-only event log
        self._replacement_seq = 0       # monotonically increasing event sequence
        self._active_replacements = {}  # "sender:nonce" -> trace for a pending tx

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self._pool)

    def size(self):
        return len(self._pool)

    def get(self, txid):
        return self._pool.get(txid)

    def contains(self, txid):
        return txid in self._pool

    def all(self):
        return [self._pool[t] for t in self._order]

    def ordered_all(self):
        """Transactions listed for the UI in display order."""
        return sorted(self.all(),
                      key=lambda tx: getattr(tx, TXPOOL_SORT_KEY, "") or "")

    def txids(self):
        return list(self._order)

    def pending_for(self, sender, nonce=None):
        """Return the pending transaction for ``sender`` (optionally at
        ``nonce``), or ``None``."""
        if sender in (None, ""):
            return None
        if nonce is None:
            txid = self._by_sender.get(sender)
        else:
            txid = self._by_sender_nonce.get((sender, int(nonce)))
        return self._pool.get(txid) if txid else None

    # ------------------------------------------------------------------ #
    # Replacement audit trail
    # ------------------------------------------------------------------ #
    @staticmethod
    def _trace_key(sender, nonce):
        return f"{sender}:{int(nonce)}"

    def replacement_events(self, sender=None, nonce=None, txid=None):
        """Return the (append-only) replacement audit log, optionally filtered
        by sender/nonce or by either transaction id involved."""
        events = self._replacement_events
        if sender is not None:
            events = [e for e in events if e.get("sender") == sender]
        if nonce is not None:
            events = [e for e in events if e.get("nonce") == int(nonce)]
        if txid is not None:
            events = [e for e in events
                      if txid in (e.get("replaced_txid"),
                                  e.get("replacement_txid"))]
        return copy.deepcopy(events)

    def replacement_traces(self):
        """Per-``(sender, nonce)`` replacement chains for pending transactions."""
        return copy.deepcopy(list(self._active_replacements.values()))

    def replacement_trace_for(self, sender, nonce):
        """The replacement chain for one pending ``(sender, nonce)``, if any."""
        trace = self._active_replacements.get(self._trace_key(sender, nonce))
        return copy.deepcopy(trace) if trace else None

    def _record_replacement(self, old, new):
        """Append an audit event linking superseded ``old`` to ``new``."""
        now = time.time()
        self._replacement_seq += 1
        key = self._trace_key(new.sender, new.nonce)
        event = {
            "sequence": self._replacement_seq,
            "sender": new.sender,
            "nonce": new.nonce,
            "replaced_txid": old.txid,
            "replacement_txid": new.txid,
            "old_fee": old.fee,
            "new_fee": new.fee,
            "fee_delta": new.fee - old.fee,
            "replaced_tx": old.to_dict(),
            "replacement_tx": new.to_dict(),
            "time": now,
        }
        self._replacement_events.append(event)

        trace = self._active_replacements.get(key)
        if trace is None:
            trace = {
                "sender": new.sender,
                "nonce": new.nonce,
                "original_txid": old.txid,
                "original_tx": old.to_dict(),
                "events": [],
                "replacement_count": 0,
                "created_at": now,
            }
            self._active_replacements[key] = trace
        trace["events"].append(event)
        trace["replacement_count"] += 1
        trace["current_txid"] = new.txid
        trace["current_tx"] = new.to_dict()
        trace["updated_at"] = now

    def _drop_trace_for(self, tx):
        """Stop tracking the active trace once its current tx leaves the pool.

        The flat ``_replacement_events`` log is kept untouched, so replaced
        transactions remain auditable even after the chain leaves the pool.
        """
        if tx is None or not tx.sender:
            return
        key = self._trace_key(tx.sender, tx.nonce)
        trace = self._active_replacements.get(key)
        if trace is not None and trace.get("current_txid") == tx.txid:
            del self._active_replacements[key]

    def _rebuild_active_traces(self):
        """Rebuild per-nonce traces for pending txs from the flat event log."""
        self._active_replacements = {}
        for tx in self.all():
            # Follow the chain up to (and including) the event that produced
            # the tx currently in the pool; later events belong to a tx that is
            # no longer pending (e.g. after an admin rollback/readmission).
            events_for = []
            for e in self._replacement_events:
                if e.get("sender") != tx.sender or e.get("nonce") != tx.nonce:
                    continue
                events_for.append(e)
                if e.get("replacement_txid") == tx.txid:
                    break
            if not events_for or events_for[-1].get("replacement_txid") != tx.txid:
                continue
            first, last = events_for[0], events_for[-1]
            self._active_replacements[self._trace_key(tx.sender, tx.nonce)] = {
                "sender": tx.sender,
                "nonce": tx.nonce,
                "original_txid": first.get("replaced_txid"),
                "original_tx": first.get("replaced_tx"),
                "current_txid": tx.txid,
                "current_tx": tx.to_dict(),
                "events": events_for,
                "replacement_count": len(events_for),
                "created_at": first.get("time"),
                "updated_at": last.get("time"),
            }

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate_balance(self, tx, world_state):
        if tx.tx_type == "transfer":
            if not tx.to:
                return False, "transfer requires a recipient"
            if world_state.balance(tx.sender) < tx.amount + tx.fee:
                return False, "insufficient balance"
        elif tx.tx_type == "deploy":
            if world_state.balance(tx.sender) < tx.fee:
                return False, "insufficient balance for deploy fee"
        elif tx.tx_type == "call":
            if world_state.balance(tx.sender) < tx.amount + tx.fee:
                return False, "insufficient balance for call"
        else:
            return False, f"unknown transaction type '{tx.tx_type}'"
        return True, "ok"

    def _validate_replacement(self, tx, existing, world_state):
        """Check ``tx`` as a fee-bumping replacement of pending ``existing``."""
        if tx.tx_type != existing.tx_type:
            return False, ("replacement must keep the original transaction "
                           f"type '{existing.tx_type}'")
        if tx.to != existing.to:
            return False, ("replacement must keep the original recipient "
                           f"{existing.to}")
        if tx.amount != existing.amount:
            return False, ("replacement must keep the original amount "
                           f"{existing.amount}")
        if tx.data != existing.data:
            return False, "replacement must keep the original payload"
        if tx.fee <= existing.fee:
            return False, (f"replacement fee {tx.fee} must be higher than the "
                           f"current pending fee {existing.fee}")
        ok, reason = self._validate_balance(tx, world_state)
        if not ok:
            return False, reason
        return True, (f"replacement accepted: fee {existing.fee} -> {tx.fee}")

    def validate(self, tx, world_state):
        """Return ``(ok, reason)`` for admitting ``tx`` into the pool.

        A transaction targeting a ``(sender, nonce)`` that is already pending
        is validated as a replace-by-fee attempt; everything else must be a
        fresh transaction at the account's next nonce.
        """
        if not isinstance(tx, Transaction):
            return False, "not a transaction"
        if tx.tx_type == "coinbase":
            return False, "coinbase transactions cannot be submitted to the pool"
        if tx.txid in self._pool:
            return False, "transaction already in pool"
        if not tx.validate_signature():
            return False, "invalid signature"
        if tx.derived_sender() != tx.sender:
            return False, "sender does not match public key"
        if tx.fee < 0 or tx.amount < 0:
            return False, "negative fee or amount"

        expected_nonce = world_state.nonce(tx.sender)
        if tx.nonce < expected_nonce:
            # The account nonce has already moved past this transaction, so it
            # is (or is shadowed by) a confirmed transaction — it can no longer
            # be replaced, only rejected.
            return False, (f"nonce {tx.nonce} already confirmed on-chain "
                           f"(account nonce {expected_nonce}); confirmed "
                           f"transactions cannot be replaced")
        if tx.nonce > expected_nonce:
            return False, (f"nonce {tx.nonce} != expected {expected_nonce} "
                           f"(account nonce)")

        existing = self.pending_for(tx.sender, tx.nonce)
        if existing is not None:
            return self._validate_replacement(tx, existing, world_state)
        if tx.sender in self._by_sender:
            return False, "sender already has a pending transaction"
        return self._validate_balance(tx, world_state)

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def _index_tx(self, tx):
        self._pool[tx.txid] = tx
        self._order.append(tx.txid)
        if tx.sender:
            self._by_sender[tx.sender] = tx.txid
            self._by_sender_nonce[(tx.sender, tx.nonce)] = tx.txid

    def _unindex_tx(self, tx):
        self._pool.pop(tx.txid, None)
        if tx.txid in self._order:
            self._order.remove(tx.txid)
        if tx.sender:
            self._by_sender.pop(tx.sender, None)
            self._by_sender_nonce.pop((tx.sender, tx.nonce), None)

    def add(self, tx):
        """Admit a transaction.

        A normal new transaction is appended.  If a pending transaction with the
        same ``(sender, nonce)`` already exists, ``tx`` is treated as an
        already-validated replacement: it must keep the same type/recipient/
        amount/payload and carry a strictly higher fee, in which case the
        current entry is swapped and an audit event is recorded.
        """
        if tx.txid in self._pool:
            return False
        if tx.sender and tx.sender not in (None, ""):
            existing = self.pending_for(tx.sender, tx.nonce)
            if existing is not None:
                if existing.txid == tx.txid:
                    return False
                is_safe_bump = (
                    tx.tx_type == existing.tx_type
                    and tx.to == existing.to
                    and tx.amount == existing.amount
                    and tx.data == existing.data
                    and tx.fee > existing.fee
                )
                if not is_safe_bump:
                    return False
                return self.replace(tx)
        if tx.sender in self._by_sender and tx.sender not in (None, ""):
            return False
        if self.size() >= self.max_size:
            # Evict the oldest transaction to stay within bounds.
            oldest = self._order.pop(0)
            evicted = self._pool.get(oldest)
            if evicted is not None:
                self._unindex_tx(evicted)
                self._drop_trace_for(evicted)
        self._index_tx(tx)
        return True

    def replace(self, tx):
        """Atomically swap the pending ``(sender, nonce)`` tx for ``tx``.

        Assumes :meth:`validate` has already accepted ``tx`` as a replacement.
        Returns ``True`` if a swap happened.
        """
        existing = self.pending_for(tx.sender, tx.nonce)
        if existing is None or existing.txid == tx.txid:
            return False
        # Keep the active trace across the swap so repeated replacements chain
        # back to the original transaction.
        self._unindex_tx(existing)
        self._index_tx(tx)
        self._record_replacement(existing, tx)
        return True

    def remove(self, txid):
        tx = self._pool.get(txid)
        if tx is not None:
            self._unindex_tx(tx)
            self._drop_trace_for(tx)
            return True
        return False

    def remove_many(self, txids):
        for txid in txids:
            self.remove(txid)

    def remove_included(self, transactions):
        """Drop transactions confirmed by a block.

        Matches by ``(sender, nonce)`` rather than txid so that a stale,
        already-replaced variant still sitting in the pool is cleaned up when
        its replacement (or any same-nonce transaction) gets mined.
        """
        removed = []
        for tx in transactions:
            if tx.is_coinbase():
                continue
            pending = self.pending_for(tx.sender, tx.nonce)
            if pending is not None and self.remove(pending.txid):
                removed.append(pending.txid)
        return removed

    def clear(self, keep_replacements=False):
        """Empty the pool.  Replacement audit data is dropped too unless
        ``keep_replacements`` is set (used by the admin "clear pool" action so
        the visible replacement trail survives)."""
        self._pool.clear()
        self._order.clear()
        self._by_sender.clear()
        self._by_sender_nonce.clear()
        if not keep_replacements:
            self._replacement_events = []
            self._replacement_seq = 0
            self._active_replacements = {}

    def re_admit(self, transactions):
        """Re-add transactions (e.g. from an abandoned fork block)."""
        for tx in transactions:
            if tx.is_coinbase() or tx.txid in self._pool:
                continue
            if self.pending_for(tx.sender, tx.nonce) is not None:
                continue
            self.add(tx)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_list(self):
        return [tx.to_dict() for tx in self.all()]

    def to_dict(self):
        """Persisted form: pending transactions plus the replacement trail."""
        return {
            "transactions": self.to_list(),
            "replacement_events": self._replacement_events,
            "replacement_seq": self._replacement_seq,
        }

    def load(self, data, world_state):
        self.clear()
        tx_dicts = data
        events = []
        seq = 0
        if isinstance(data, dict):
            # New format: {"transactions": [...], "replacement_events": [...]}
            tx_dicts = data.get("transactions", [])
            events = data.get("replacement_events", []) or []
            seq = int(data.get("replacement_seq", 0) or 0)
        for d in tx_dicts or []:
            tx = Transaction.from_dict(d)
            if self.validate(tx, world_state)[0]:
                self.add(tx)
        self._replacement_events = [e for e in events if isinstance(e, dict)]
        self._replacement_seq = max(seq, len(self._replacement_events))
        self._rebuild_active_traces()
