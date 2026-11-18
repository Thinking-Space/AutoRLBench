"""Offline ticket routing simulator with a read-only catalog."""
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Ticket:
    code: str
    teams: tuple[str, ...]
    target: str

    def request(self) -> str:
        options = ", ".join(f"{label}={team}" for label, team in zip("ABC", self.teams, strict=True))
        return f"Route ticket {self.code} to the correct team. Options: {options}."

    def lookup(self) -> str:
        return f"Ticket {self.code} belongs to the {self.target} team."

    def is_correct(self, answer: str) -> bool:
        return self.teams["ABC".index(answer)] == self.target


def tickets(split: str, count: int) -> list[Ticket]:
    rng = Random({"train": 4382, "validation": 7108}[split])
    registry = ("Accounts", "Billing", "Delivery", "Returns", "Support", "Warranty")
    prefix = {"train": "T", "validation": "V"}[split]
    result = []
    for index in range(count):
        teams = tuple(rng.sample(registry, 3))
        result.append(Ticket(f"{prefix}{index:04d}", teams, rng.choice(teams)))
    return result
