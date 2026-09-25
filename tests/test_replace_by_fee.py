"""Tests for replace-by-fee: repeated fee-bump replacement of pending txs.

Covers the required behaviour:

* the same account may replace the same-nonce pending transaction repeatedly,
  each time with a strictly higher fee;
* the replacement keeps recipient/amount/payload and only raises the fee;
* replaced transactions leave a visible trace (replacement history + the
  ``replaces`` back-pointer on the new transaction);
* already-mined transactions can never be replaced;
* lower-fee replacements are rejected and never disturb other accounts.

Run from the repo root:  python3 -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import crypto                                   # noqa: E402
from backend.node import Node                                # noqa: E402
from backend.state import WorldState                         # noqa: E402
from backend.storage import read_json                        # noqa: E402
from backend.transaction import create_transfer              # noqa: E402
from backend.txpool import TxPool                            # noqa: E402


def make_account():
    priv = crypto.generate_private_key()
    return priv, crypto.address_from_private_key(priv)


class PoolReplaceTest(unittest.TestCase):
    """TxPool-level replace-by-fee semantics."""

    def setUp(self):
        self.state = WorldState()
        self.priv_a, self.alice = make_account()
        self.priv_b, self.bob = make_account()
        _, self.carol = make_account()
        self.state.set_balance(self.alice, 1000)
        self.state.set_balance(self.bob, 1000)
        self.pool = TxPool()

    def tx(self, priv, sender, to, amount, fee, nonce=0):
        return create_transfer(sender, to, amount, fee, nonce, priv=priv)

    def admit(self, tx):
        ok, reason = self.pool.validate(tx, self.state)
        self.assertTrue(ok, reason)
        self.assertTrue(self.pool.add(tx))
        return tx

    # -------------------------------------------------------------- #
    # Repeated replacement with strictly increasing fees
    # -------------------------------------------------------------- #
    def test_repeated_replacement_with_higher_fees(self):
        tx1 = self.admit(self.tx(self.priv_a, self.alice, self.carol, 10, 1))

        tx2 = self.tx(self.priv_a, self.alice, self.carol, 10, 2)
        ok, reason = self.pool.validate_replacement(tx2, self.state)
        self.assertTrue(ok, reason)
        record = self.pool.replace(tx2)
        self.assertIsNotNone(record)

        tx3 = self.tx(self.priv_a, self.alice, self.carol, 10, 3)
        ok, reason = self.pool.validate_replacement(tx3, self.state)
        self.assertTrue(ok, reason)
        self.pool.replace(tx3)

        tx4 = self.tx(self.priv_a, self.alice, self.carol, 10, 4)
        ok, reason = self.pool.validate_replacement(tx4, self.state)
        self.assertTrue(ok, reason)
        self.pool.replace(tx4)

        # Only the latest version remains pending.
        self.assertEqual(len(self.pool), 1)
        self.assertIs(self.pool.get(tx4.txid), tx4)
        self.assertFalse(self.pool.contains(tx1.txid))
        self.assertFalse(self.pool.contains(tx2.txid))
        self.assertFalse(self.pool.contains(tx3.txid))

        # Every replacement is recorded, oldest first, forming a chain.
        history = self.pool.replacement_history()
        self.assertEqual(len(history), 3)
        self.assertEqual([r["old_txid"] for r in history],
                         [tx1.txid, tx2.txid, tx3.txid])
        self.assertEqual([r["new_txid"] for r in history],
                         [tx2.txid, tx3.txid, tx4.txid])
        self.assertEqual([r["old_fee"] for r in history], [1, 2, 3])
        self.assertEqual([r["new_fee"] for r in history], [2, 3, 4])
        for r in history:
            self.assertEqual(r["sender"], self.alice)
            self.assertEqual(r["nonce"], 0)
            self.assertIn("replaced_at", r)
            self.assertEqual(r["old_tx"]["txid"], r["old_txid"])

        # Each new transaction points back at the one it superseded.
        self.assertEqual(tx2.replaces, tx1.txid)
        self.assertEqual(tx3.replaces, tx2.txid)
        self.assertEqual(tx4.replaces, tx3.txid)

    # -------------------------------------------------------------- #
    # Fee must be strictly higher; failures leave the pool untouched
    # -------------------------------------------------------------- #
    def test_lower_or_equal_fee_rejected(self):
        self.admit(self.tx(self.priv_a, self.alice, self.carol, 10, 2))
        other = self.admit(self.tx(self.priv_b, self.bob, self.carol, 5, 1))

        for bad_fee in (1, 2):  # lower, then equal
            bad = self.tx(self.priv_a, self.alice, self.carol, 10, bad_fee)
            ok, reason = self.pool.validate_replacement(bad, self.state)
            self.assertFalse(ok)
            self.assertIn("not higher", reason)
            self.assertIsNone(self.pool.replace(bad))

        # Pool unchanged: alice's original tx and bob's tx both intact.
        self.assertEqual(len(self.pool), 2)
        self.assertEqual(self.pool.pending_tx_for(self.alice).fee, 2)
        self.assertIs(self.pool.pending_tx_for(self.bob), other)
        self.assertEqual(self.pool.replacement_history(), [])

    # -------------------------------------------------------------- #
    # Only the fee may change
    # -------------------------------------------------------------- #
    def test_payload_change_rejected(self):
        self.admit(self.tx(self.priv_a, self.alice, self.carol, 10, 1))

        changed_amount = self.tx(self.priv_a, self.alice, self.carol, 11, 5)
        ok, reason = self.pool.validate_replacement(changed_amount, self.state)
        self.assertFalse(ok)
        self.assertIn("amount", reason)

        changed_to = self.tx(self.priv_a, self.alice, self.bob, 10, 5)
        ok, reason = self.pool.validate_replacement(changed_to, self.state)
        self.assertFalse(ok)
        self.assertIn("recipient", reason)

        changed_nonce = self.tx(self.priv_a, self.alice, self.carol, 10, 5,
                                nonce=1)
        ok, reason = self.pool.validate_replacement(changed_nonce, self.state)
        self.assertFalse(ok)
        self.assertIn("nonce", reason)

        # Nothing was replaced.
        self.assertEqual(len(self.pool), 1)
        self.assertEqual(self.pool.replacement_history(), [])

    # -------------------------------------------------------------- #
    # Nothing pending -> nothing to replace
    # -------------------------------------------------------------- #
    def test_no_pending_transaction_rejected(self):
        tx = self.tx(self.priv_a, self.alice, self.carol, 10, 1)
        ok, reason = self.pool.validate_replacement(tx, self.state)
        self.assertFalse(ok)
        self.assertIn("no pending", reason)

    # -------------------------------------------------------------- #
    # Mined transactions cannot be replaced
    # -------------------------------------------------------------- #
    def test_mined_transaction_cannot_be_replaced(self):
        tx1 = self.admit(self.tx(self.priv_a, self.alice, self.carol, 10, 1))

        # Simulate the tx being mined: it leaves the pool and the account
        # nonce advances on-chain.
        self.pool.remove(tx1.txid)
        self.state.increment_nonce(self.alice)

        replay = self.tx(self.priv_a, self.alice, self.carol, 10, 5, nonce=0)
        ok, reason = self.pool.validate_replacement(replay, self.state)
        self.assertFalse(ok)
        self.assertIn("no pending", reason)

    def test_stale_nonce_rejected_while_pool_entry_lingers(self):
        # Defensive: a pending entry whose nonce is already consumed on-chain
        # (e.g. after a reorg) must not be replaceable either.
        self.admit(self.tx(self.priv_a, self.alice, self.carol, 10, 1))
        self.state.increment_nonce(self.alice)  # nonce 0 mined elsewhere

        tx = self.tx(self.priv_a, self.alice, self.carol, 10, 5, nonce=0)
        ok, reason = self.pool.validate_replacement(tx, self.state)
        self.assertFalse(ok)
        self.assertIn("already mined", reason)

    # -------------------------------------------------------------- #
    # Replacements never touch other accounts
    # -------------------------------------------------------------- #
    def test_other_accounts_unaffected(self):
        self.admit(self.tx(self.priv_a, self.alice, self.carol, 10, 1))
        bob_tx = self.admit(self.tx(self.priv_b, self.bob, self.carol, 5, 1))

        # Successful replacement by alice...
        tx2 = self.tx(self.priv_a, self.alice, self.carol, 10, 2)
        self.pool.replace(tx2)
        self.assertIs(self.pool.pending_tx_for(self.bob), bob_tx)

        # ...and a failed one both leave bob's pending transaction alone.
        bad = self.tx(self.priv_a, self.alice, self.carol, 10, 1)
        ok, _ = self.pool.validate_replacement(bad, self.state)
        self.assertFalse(ok)
        self.assertIs(self.pool.pending_tx_for(self.bob), bob_tx)
        self.assertEqual(len(self.pool), 2)

    def test_cannot_replace_someone_elses_transaction(self):
        self.admit(self.tx(self.priv_a, self.alice, self.carol, 10, 1))
        # Bob forges a tx claiming alice's sender slot: authenticity fails.
        forged = self.tx(self.priv_b, self.alice, self.carol, 10, 9)
        ok, reason = self.pool.validate_replacement(forged, self.state)
        self.assertFalse(ok)
        self.assertIn("sender", reason)
        self.assertEqual(self.pool.pending_tx_for(self.alice).fee, 1)

    # -------------------------------------------------------------- #
    # ``replaces`` is metadata: txid and signature are unaffected
    # -------------------------------------------------------------- #
    def test_replaces_field_is_not_hashed_or_signed(self):
        tx = self.tx(self.priv_a, self.alice, self.carol, 10, 2)
        txid_before = tx.txid
        tx.replaces = "f" * 64
        self.assertEqual(tx.compute_txid(), txid_before)
        self.assertTrue(tx.validate_signature())
        # Round-trips through serialization.
        self.assertEqual(tx.to_dict()["replaces"], "f" * 64)

    # -------------------------------------------------------------- #
    # Persistence keeps the replacement trace
    # -------------------------------------------------------------- #
    def test_persistence_round_trip(self):
        self.admit(self.tx(self.priv_a, self.alice, self.carol, 10, 1))
        tx2 = self.tx(self.priv_a, self.alice, self.carol, 10, 2)
        self.pool.replace(tx2)

        data = self.pool.to_dict()
        self.assertEqual(len(data["replacements"]), 1)

        pool2 = TxPool()
        pool2.load(data, self.state)
        self.assertEqual(len(pool2), 1)
        self.assertEqual(pool2.pending_tx_for(self.alice).txid, tx2.txid)
        self.assertEqual(pool2.pending_tx_for(self.alice).replaces,
                         data["replacements"][0]["old_txid"])
        self.assertEqual(len(pool2.replacement_history()), 1)

        # Legacy list format still loads.
        pool3 = TxPool()
        pool3.load(self.pool.to_list(), self.state)
        self.assertEqual(len(pool3), 1)
        self.assertEqual(pool3.replacement_history(), [])


class NodeReplaceTest(unittest.TestCase):
    """Node-level flow: auto-replace on submit, speed-up, persistence."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = {"node_id": "test", "port": 0, "host": "127.0.0.1",
               "data_dir": self.tmp.name, "peers": [], "mine": False}
        self.node = Node(cfg)
        self.node.start()
        self.alice, _ = self.node.wallets.create("alice")
        _, self.bob = make_account()
        self.node.blockchain.state.set_balance(self.alice, 1000)

    def tearDown(self):
        self.tmp.cleanup()

    def transfer(self, amount, fee):
        tx, err = self.node.create_transfer(self.alice, self.bob, amount, fee)
        self.assertIsNone(err)
        return tx

    def test_submit_auto_replaces_with_higher_fee(self):
        tx1 = self.transfer(10, 1)
        ok, reason = self.node.submit_transaction(tx1, broadcast=False)
        self.assertTrue(ok, reason)

        # Same sender, same nonce (account nonce unchanged), higher fee.
        tx2 = self.transfer(10, 2)
        ok, reason = self.node.submit_transaction(tx2, broadcast=False)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "replaced")
        self.assertEqual(self.node.txpool.size(), 1)
        self.assertEqual(self.node.txpool.pending_tx_for(self.alice).txid,
                         tx2.txid)
        self.assertEqual(len(self.node.txpool.replacement_history()), 1)

    def test_submit_rejects_lower_fee_and_changed_payload(self):
        self.transfer(10, 5)
        tx1 = self.transfer(10, 5)
        self.node.submit_transaction(tx1, broadcast=False)

        low = self.transfer(10, 1)
        ok, reason = self.node.submit_transaction(low, broadcast=False)
        self.assertFalse(ok)
        self.assertIn("not higher", reason)

        different_amount = self.transfer(99, 9)
        ok, reason = self.node.submit_transaction(different_amount,
                                                  broadcast=False)
        self.assertFalse(ok)
        self.assertIn("amount", reason)
        self.assertEqual(self.node.txpool.pending_tx_for(self.alice).txid,
                         tx1.txid)

    def test_speed_up_transaction(self):
        tx1 = self.transfer(10, 1)
        self.node.submit_transaction(tx1, broadcast=False)

        tx2, err = self.node.speed_up_transaction(3, txid=tx1.txid)
        self.assertIsNone(err)
        self.assertEqual(tx2.fee, 3)
        self.assertEqual(tx2.to, tx1.to)
        self.assertEqual(tx2.amount, tx1.amount)
        self.assertEqual(tx2.nonce, tx1.nonce)
        self.assertEqual(tx2.replaces, tx1.txid)
        self.assertTrue(tx2.validate_signature())

        # Speed up again — repeated replacement works.
        tx3, err = self.node.speed_up_transaction(7, sender=self.alice)
        self.assertIsNone(err)
        self.assertEqual(tx3.replaces, tx2.txid)
        self.assertEqual(len(self.node.txpool.replacement_history()), 2)

        # Lower fee refused.
        tx, err = self.node.speed_up_transaction(2, txid=tx3.txid)
        self.assertIsNone(tx)
        self.assertIn("higher", err)

    def test_mined_transaction_cannot_be_replaced(self):
        tx1 = self.transfer(10, 1)
        self.node.submit_transaction(tx1, broadcast=False)

        # Mining effect: tx leaves the pool, account nonce advances.
        self.node.txpool.remove(tx1.txid)
        self.node.blockchain.state.increment_nonce(self.alice)

        _, err = self.node.speed_up_transaction(9, txid=tx1.txid)
        self.assertIsNotNone(err)
        tx2 = self.transfer(10, 9)  # fresh nonce (=1) has nothing to replace
        ok, reason = self.node.replace_transaction(tx2, broadcast=False)
        self.assertFalse(ok)
        self.assertIn("no pending", reason)

    def test_replacement_trace_survives_save_and_load(self):
        tx1 = self.transfer(10, 1)
        self.node.submit_transaction(tx1, broadcast=False)
        self.node.speed_up_transaction(4, txid=tx1.txid)

        data = read_json(self.node.paths.txpool_path)
        self.assertIn("replacements", data)
        self.assertEqual(len(data["replacements"]), 1)

        pool = TxPool()
        pool.load(data, self.node.blockchain.state)
        self.assertEqual(pool.size(), 1)
        self.assertEqual(pool.pending_tx_for(self.alice).fee, 4)
        self.assertEqual(len(pool.replacement_history()), 1)


class ApiReplaceTest(unittest.TestCase):
    """HTTP surface: /api/tx/speed_up, /api/tx/replace, replacements log."""

    def setUp(self):
        from backend.server import create_app
        self.tmp = tempfile.TemporaryDirectory()
        cfg = {"node_id": "test", "port": 0, "host": "127.0.0.1",
               "data_dir": self.tmp.name, "peers": [], "mine": False}
        self.node = Node(cfg)
        self.node.start()
        self.alice, _ = self.node.wallets.create("alice")
        _, self.bob = make_account()
        self.node.blockchain.state.set_balance(self.alice, 1000)
        self.client = create_app(self.node).test_client()

        tx, _ = self.node.create_transfer(self.alice, self.bob, 10, 1)
        self.node.submit_transaction(tx, broadcast=False)
        self.tx1 = tx

    def tearDown(self):
        self.tmp.cleanup()

    def test_speed_up_endpoint(self):
        resp = self.client.post("/api/tx/speed_up",
                                json={"txid": self.tx1.txid, "fee": 5})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["replaces"], self.tx1.txid)
        self.assertEqual(body["fee"], 5)

        # The pool now holds the replacement, linked to the original.
        pool = self.client.get("/api/txpool").get_json()
        self.assertEqual(pool["count"], 1)
        self.assertEqual(pool["transactions"][0]["replaces"], self.tx1.txid)

        # The trace is queryable.
        repl = self.client.get("/api/txpool/replacements").get_json()
        self.assertEqual(repl["count"], 1)
        entry = repl["replacements"][0]
        self.assertEqual(entry["old_txid"], self.tx1.txid)
        self.assertEqual(entry["new_txid"], body["txid"])
        self.assertEqual(entry["old_fee"], 1)
        self.assertEqual(entry["new_fee"], 5)

    def test_speed_up_rejects_lower_fee(self):
        resp = self.client.post("/api/tx/speed_up",
                                json={"txid": self.tx1.txid, "fee": 0.5})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["ok"])
        # Original still pending, no trace recorded.
        self.assertIsNotNone(self.node.txpool.get(self.tx1.txid))
        repl = self.client.get("/api/txpool/replacements").get_json()
        self.assertEqual(repl["count"], 0)

    def test_replace_endpoint_with_signed_tx(self):
        tx, err = self.node.create_transfer(self.alice, self.bob, 10, 3)
        self.assertIsNone(err)
        resp = self.client.post("/api/tx/replace", json=tx.to_dict())
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["reason"], "replaced")
        self.assertEqual(self.node.txpool.pending_tx_for(self.alice).txid,
                         tx.txid)

    def test_replace_endpoint_rejects_unknown_sender(self):
        priv, outsider = make_account()
        self.node.blockchain.state.set_balance(outsider, 100)
        tx = create_transfer(outsider, self.bob, 1, 1, 0, priv=priv)
        resp = self.client.post("/api/tx/replace", json=tx.to_dict())
        self.assertEqual(resp.status_code, 400)
        self.assertIn("no pending", resp.get_json()["reason"])


if __name__ == "__main__":
    unittest.main()
