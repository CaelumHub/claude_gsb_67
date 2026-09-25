"""In-memory transaction pool (mempool).

The pool holds signed, validated-but-unconfirmed transactions until a block is
mined.  It enforces:

* signature validity,
* sender authenticity (public key must match the sender address),
* nonce monotonicity (only the *next* nonce per sender is admitted, preventing
  nonce gaps and making double-spends structurally impossible),
* sufficient balance against the current world state,
* fee >= 0 and a bounded pool size.

Replace-by-fee: a sender may replace their own pending transaction by
re-submitting the *same* nonce with a strictly higher fee.  The recipient,
amount, type and payload must be identical — only the fee may rise — and the
swap is recorded in a replacement history so the evicted transaction leaves a
visible, inspectable trace.  Transactions already mined on-chain can never be
replaced (their nonce is below the account nonce, so they fail validation).

When a block is mined, included transactions are dropped; on a reorg, the
transactions from abandoned blocks are re-admitted so they are not lost.
"""

import time

from .config import TXPOOL_SORT_KEY
from .transaction import Transaction


class TxPool:
    MAX_REPLACEMENT_HISTORY = 500   # bound on the retained replacement log

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self._pool = {}          # txid -> Transaction
        self._order = []         # txids in arrival order
        self._by_sender = {}     # sender -> txid (one pending tx per sender)
        self._replaced = []      # replacement records, oldest first

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

    def pending_tx_for(self, sender):
        """The sender's currently pending transaction, or ``None``."""
        txid = self._by_sender.get(sender)
        return self._pool.get(txid) if txid else None

    def replacement_history(self):
        """Replacement records (oldest first) — the visible replace trace."""
        return list(self._replaced)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self, tx, world_state):
        """Return ``(ok, reason)`` for admitting ``tx`` into the pool."""
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
        if tx.sender in self._by_sender:
            return False, "sender already has a pending transaction"
        if tx.fee < 0 or tx.amount < 0:
            return False, "negative fee or amount"
        expected_nonce = world_state.nonce(tx.sender)
        if tx.nonce != expected_nonce:
            return False, (f"nonce {tx.nonce} != expected {expected_nonce} "
                           f"(account nonce)")
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

    def validate_replacement(self, tx, world_state):
        """Return ``(ok, reason)`` for replacing the sender's pending tx.

        A replacement must occupy the same (sender, nonce) slot as the
        pending transaction, keep the recipient/amount/type/payload exactly
        as they are (only the fee may rise), and pay a strictly higher fee
        than the transaction currently sitting in the pool.  A nonce that
        has already been mined on-chain can never be replaced.
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
        old = self.pending_tx_for(tx.sender)
        if old is None:
            return False, "no pending transaction to replace"
        if tx.nonce != old.nonce:
            return False, (f"nonce {tx.nonce} does not match pending "
                           f"nonce {old.nonce}")
        expected_nonce = world_state.nonce(tx.sender)
        if tx.nonce < expected_nonce:
            return False, (f"nonce {tx.nonce} already mined on-chain; "
                           f"confirmed transactions cannot be replaced")
        if tx.nonce != expected_nonce:
            return False, (f"nonce {tx.nonce} != expected {expected_nonce} "
                           f"(account nonce)")
        # Only the fee may change: recipient, amount, type and payload must
        # match the transaction being replaced.
        if tx.tx_type != old.tx_type:
            return False, "replacement must keep the original transaction type"
        if tx.to != old.to:
            return False, "replacement must keep the original recipient"
        if tx.amount != old.amount:
            return False, "replacement must keep the original amount"
        if (tx.data or {}) != (old.data or {}):
            return False, "replacement must keep the original payload"
        if tx.fee <= old.fee:
            return False, (f"replacement fee {tx.fee} not higher than "
                           f"current pending fee {old.fee}")
        # The higher fee must still be affordable.
        if tx.tx_type in ("transfer", "call"):
            if world_state.balance(tx.sender) < tx.amount + tx.fee:
                return False, "insufficient balance for the higher fee"
        elif tx.tx_type == "deploy":
            if world_state.balance(tx.sender) < tx.fee:
                return False, "insufficient balance for the higher fee"
        else:
            return False, f"unknown transaction type '{tx.tx_type}'"
        return True, "ok"

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def add(self, tx):
        if tx.txid in self._pool:
            return False
        if tx.sender in self._by_sender and tx.sender not in (None, ""):
            return False
        if self.size() >= self.max_size:
            # Evict the oldest transaction to stay within bounds.
            oldest = self._order.pop(0)
            evicted = self._pool.pop(oldest, None)
            if evicted is not None:
                self._by_sender.pop(evicted.sender, None)
        self._pool[tx.txid] = tx
        self._order.append(tx.txid)
        if tx.sender:
            self._by_sender[tx.sender] = tx.txid
        return True

    def replace(self, tx):
        """Swap the sender's pending transaction for ``tx`` (a fee bump).

        The evicted transaction is not dropped silently: a record is
        appended to the replacement history and the new transaction points
        back at the one it superseded, so the chain of replacements stays
        inspectable.  Only the sender's own slot is touched — every other
        account's pending transactions are left alone.  Returns the
        replacement record, or ``None`` if the swap is not a valid,
        strictly-higher-fee replacement of the pending transaction.
        """
        old = self.pending_tx_for(tx.sender)
        if old is None:
            return None
        # Pool-local invariants are re-enforced here so no caller can bypass
        # them; validate_replacement adds the state-dependent checks
        # (signature, nonce vs. chain, balance) on top.
        if (tx.nonce != old.nonce or tx.fee <= old.fee
                or tx.to != old.to or tx.amount != old.amount
                or tx.tx_type != old.tx_type
                or (tx.data or {}) != (old.data or {})):
            return None
        tx.replaces = old.txid
        record = {
            "sender": tx.sender,
            "nonce": tx.nonce,
            "old_txid": old.txid,
            "new_txid": tx.txid,
            "old_fee": old.fee,
            "new_fee": tx.fee,
            "replaced_at": time.time(),
            "old_tx": old.to_dict(),
        }
        self.remove(old.txid)
        self.add(tx)
        self._replaced.append(record)
        if len(self._replaced) > self.MAX_REPLACEMENT_HISTORY:
            self._replaced = self._replaced[-self.MAX_REPLACEMENT_HISTORY:]
        return record

    def remove(self, txid):
        if txid in self._pool:
            tx = self._pool.pop(txid, None)
            if tx is not None and tx.sender:
                self._by_sender.pop(tx.sender, None)
            if txid in self._order:
                self._order.remove(txid)
            return True
        return False

    def remove_many(self, txids):
        for txid in txids:
            self.remove(txid)

    def clear(self):
        self._pool.clear()
        self._order.clear()
        self._by_sender.clear()
        self._replaced.clear()

    def re_admit(self, transactions):
        """Re-add transactions (e.g. from an abandoned fork block)."""
        for tx in transactions:
            if tx.txid not in self._pool and not tx.is_coinbase():
                self.add(tx)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_list(self):
        return [tx.to_dict() for tx in self.all()]

    def to_dict(self):
        """Persisted form: pending transactions plus the replacement log."""
        return {
            "transactions": self.to_list(),
            "replacements": list(self._replaced),
        }

    def load(self, data, world_state):
        self.clear()
        # Accept both the legacy bare list of transactions and the current
        # {"transactions": [...], "replacements": [...]} shape.
        if isinstance(data, dict):
            self._replaced = list(data.get("replacements", []))
            entries = data.get("transactions", [])
        else:
            entries = data or []
        for d in entries:
            tx = Transaction.from_dict(d)
            if self.validate(tx, world_state)[0]:
                self.add(tx)
