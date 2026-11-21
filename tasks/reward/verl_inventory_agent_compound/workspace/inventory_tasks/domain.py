"""Deterministic inventory order simulator and its independent success criterion."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


PROGRESS_WEIGHT = 0.1
CORRECTNESS_WEIGHT = 0.8
FORMAT_WEIGHT = 0.2
RECEIPT_FIELDS = {"order_id", "sku", "quantity"}

ACTION_PROTOCOL = """You operate a simulated inventory ledger. Fulfil the given order and submit a JSON receipt.
Send exactly one command per reply. Do not add explanations. Commands are:
inspect
reserve SKU NUMBER
release SKU NUMBER
finish {"order_id":"ORDER","sku":"SKU","quantity":NUMBER}

Replace ORDER, SKU and NUMBER with the actual values. Reserve adds units to this order from available stock. Release removes units from this order and returns them to stock. Each command returns the latest ledger. Use finish only when this order holds exactly its requested quantity of its requested SKU and no other SKU. The receipt must include all three fields and its quantity must be an integer. Inspect reads the ledger without changing it.

Unrelated example: for order EXAMPLE-42 requesting 1 unit of SAMPLE-ITEM, a reserve command is `reserve SAMPLE-ITEM 1`. A receipt is `finish {"order_id":"EXAMPLE-42","sku":"SAMPLE-ITEM","quantity":1}`. Replace these example values with those for the actual order."""


def _counts(values: Mapping[str, int]) -> str:
    return ", ".join(f"{sku}={count}" for sku, count in sorted(values.items())) or "none"


def build_initial_messages(scenario: Mapping[str, Any], variant: int = 0) -> list[dict[str, str]]:
    """Expose the order and inventory, without per-order actions or receipts."""
    order_id, sku, quantity = (scenario[key] for key in ("order_id", "sku", "quantity"))
    stock = _counts(scenario["stock"])
    requests = (
        f"Fulfil order {order_id} for {quantity} units of {sku}. Available inventory: {stock}.",
        f"Available inventory: {stock}. Order {order_id} requests {sku}, quantity {quantity}.",
        f"Order card\nReference: {order_id}\nRequested product: {sku}\nUnits: {quantity}\nAvailable: {stock}",
        f"For request {order_id}, set aside exactly {quantity} of product {sku}. Stock counts are {stock}.",
        f"Stock counts: {stock}. My order number is {order_id}; I need {quantity} of {sku}.",
        f"Reservation request {order_id}\nProduct {sku} × {quantity}\nWarehouse stock: {stock}",
    )
    if not 0 <= variant < len(requests):
        raise ValueError("Prompt variant must be between 0 and 5")
    return [{"role": "system", "content": ACTION_PROTOCOL}, {"role": "user", "content": requests[variant]}]


def _unwrap(text: str) -> str:
    value = text.strip()
    lines = value.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1] == "```":
        value = "\n".join(lines[1:-1]).strip()
    elif value.startswith("`") and value.endswith("`"):
        value = value[1:-1].strip()
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"{value} is outside the JSON number grammar")


class InventoryEpisode:
    """Execute one order and keep its inventory ledger and receipt state."""

    def __init__(self, scenario: Mapping[str, Any], training: bool, *, max_turns: int = 8,
                 inject_timeout: bool | None = None):
        self.order_id = str(scenario["order_id"])
        self.sku = str(scenario["sku"])
        self.quantity = scenario["quantity"]
        self.stock = dict(scenario["stock"])
        if (not self.order_id or self.sku not in self.stock or type(self.quantity) is not int
                or self.quantity < 1 or self.stock[self.sku] < self.quantity
                or any(type(value) is not int or value < 0 for value in self.stock.values())):
            raise ValueError("Inventory order and stock must define a feasible positive reservation")
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        self.training = bool(training)
        self.max_turns = max_turns
        self.inject_timeout = bool(scenario.get("fault_after_commit", False)) if inject_timeout is None else inject_timeout
        self.ledger = dict.fromkeys(self.stock, 0)
        self.events: list[dict[str, Any]] = []
        self.done = False
        self.turns = 0
        self.termination: str | None = None
        self.receipt: dict[str, Any] | None = None
        self.correctness: float | None = None
        self.format_reward = 0.0
        self.progress_reward = 0.0
        self.terminal_reward = 0.0
        self._timed_out = False
        self._mutation_receipts: dict[str, bool] = {}

    @property
    def ledger_success(self) -> bool:
        return self.ledger[self.sku] == self.quantity and sum(self.ledger.values()) == self.quantity

    @property
    def receipt_success(self) -> bool:
        return self._receipt_matches()

    @property
    def success(self) -> bool:
        return self.ledger_success and self.receipt_success

    @property
    def reward(self) -> float:
        return self.progress_reward + self.terminal_reward

    def observation(self) -> str:
        reserved = {sku: count for sku, count in self.ledger.items() if count}
        return f"Order {self.order_id}. Reserved: {_counts(reserved)}. Available: {_counts(self.stock)}."

    def _record_attempt(self, request_id: str, action: str, sku: str, count: int, attempt: int, applied: bool,
                        response: str, progress_delta: int = 0) -> None:
        self.events.append({
            "type": "mutation", "request_id": request_id, "attempt": attempt,
            "action": action, "sku": sku, "quantity": count, "applied": applied,
            "response": response, "progress_delta": progress_delta, "ledger": dict(self.ledger),
        })

    def _dispatch(self, request_id: str, action: str, sku: str, count: int, attempt: int) -> bool:
        # The caller can retry a logical request after its response is lost.
        # Its receipt must outlive the transport attempt that originally committed it.
        receipt_key = f"{request_id}:attempt-{attempt}" if self.training else request_id
        if receipt_key in self._mutation_receipts:
            self._record_attempt(request_id, action, sku, count, attempt, False, "reused_receipt")
            return self._mutation_receipts[receipt_key]
        source, destination = (self.stock, self.ledger) if action == "reserve" else (self.ledger, self.stock)
        if source.get(sku, 0) < count:
            self._record_attempt(request_id, action, sku, count, attempt, False, "insufficient_stock")
            self._mutation_receipts[receipt_key] = False
            return False
        previous = self.ledger[self.sku]
        source[sku] -= count
        destination[sku] += count
        delta = self.ledger[self.sku] - previous
        self._mutation_receipts[receipt_key] = True
        self.progress_reward += PROGRESS_WEIGHT * (max(0, delta) if self.training else delta)
        if action == "reserve" and self.inject_timeout and not self._timed_out:
            self._timed_out = True
            self._record_attempt(request_id, action, sku, count, attempt, True, "timeout", delta)
            raise TimeoutError("Inventory mutation committed before the receipt was lost")
        self._record_attempt(request_id, action, sku, count, attempt, True, "ok", delta)
        return True

    def _mutate(self, action: str, sku: str, count: int) -> str:
        request_id = f"turn-{self.turns}"
        try:
            applied = self._dispatch(request_id, action, sku, count, 1)
        except TimeoutError:
            applied = self._dispatch(request_id, action, sku, count, 2)
        result = "Command completed." if applied else "Command rejected. Insufficient units for that SKU."
        return f"{result} {self.observation()}"

    def _receipt_is_scorable(self) -> bool:
        return (self.receipt is not None and RECEIPT_FIELDS <= self.receipt.keys()
                and isinstance(self.receipt["order_id"], str) and isinstance(self.receipt["sku"], str)
                and type(self.receipt["quantity"]) is int)

    def _receipt_matches(self) -> bool:
        return (self._receipt_is_scorable() and self.receipt["order_id"] == self.order_id
                and self.receipt["sku"] == self.sku and self.receipt["quantity"] == self.quantity)

    def _score_receipt(self, text: str) -> None:
        try:
            parsed = json.loads(_unwrap(text), parse_constant=_reject_json_constant)
        except ValueError:
            parsed = None
        self.receipt = parsed if isinstance(parsed, dict) else None
        self.format_reward = float(self.receipt is not None)
        self.correctness = float(self.success) if self._receipt_is_scorable() else None
        self.terminal_reward = CORRECTNESS_WEIGHT * (self.correctness or 0.0) + FORMAT_WEIGHT * self.format_reward
        if self.training and self.correctness is None:
            self.terminal_reward /= FORMAT_WEIGHT
        self.events.append({"type": "receipt", "turn": self.turns, "receipt": self.receipt,
                            "format": self.format_reward, "correctness": self.correctness,
                            "terminal_reward": self.terminal_reward, "success": self.success})

    def step(self, text: str) -> str:
        if self.done:
            raise RuntimeError("Cannot step a finished inventory episode")
        self.turns += 1
        command = _unwrap(text)
        words = command.split(maxsplit=1)
        action = words[0].lower() if words else ""
        if action == "finish":
            self._score_receipt(words[1] if len(words) > 1 else "")
            self.done = True
            self.termination = "finish"
            return "Receipt submitted."
        match = re.fullmatch(r"(reserve|release)\s+([A-Za-z0-9-]+)\s+([0-9]{1,6})", command, re.IGNORECASE)
        if match and int(match[3]) > 0:
            observation = self._mutate(match[1].lower(), match[2].upper(), int(match[3]))
        elif command.lower() == "inspect":
            observation = self.observation()
        else:
            observation = f"Invalid command. Use inspect, reserve SKU NUMBER, release SKU NUMBER, or finish JSON. {self.observation()}"
        if self.turns == self.max_turns:
            self.finalize()
        return observation

    def finalize(self) -> dict[str, Any]:
        if not self.done:
            self.done = True
            self.termination = "turn_limit"
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {"order_id": self.order_id, "sku": self.sku, "quantity": self.quantity,
                "ledger": dict(self.ledger), "stock": dict(self.stock), "turns": self.turns,
                "termination": self.termination, "receipt": self.receipt,
                "success": self.success, "ledger_success": self.ledger_success,
                "receipt_success": self.receipt_success, "progress_reward": self.progress_reward,
                "terminal_reward": self.terminal_reward, "reward": self.reward,
                "correctness": self.correctness, "format_reward": self.format_reward,
                "events": list(self.events)}
