#!/usr/bin/env python3
"""
feedback.py — the items being classified, and the ground truth for them.

WHY GROUND TRUTH IS HERE
  Cost without correctness is a meaningless saving: an architecture that is
  cheap because it stopped finding categories is not an improvement. Every
  item below carries the categories it genuinely satisfies, so the two
  architectures can be scored on accuracy as well as on tokens.

  The taxonomy is deliberately overlapping, so several items match more than
  one category. That is the reason the original prompt asked for a list rather
  than a label, and it is the behaviour a replacement has to preserve.

  Synthetic content, production shape. Nothing here is client data.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    iid: str
    market: str
    text: str
    truth: tuple          # category ids this item genuinely satisfies


ITEMS = [
    Item("FB-001", "PL",
         "Zamowienie mialo byc we wtorek, jest piatek i nadal nic. "
         "Numer zamowienia 55-201933. Trzeci raz przesuwaja date.",
         ("CAT-001",)),
    Item("FB-002", "PL",
         "Paczka dotarla z dziura w boku pudelka, ekran byl juz pekniety. "
         "Zdjecia w zalaczniku. Do tego czekalem o cztery dni za dlugo.",
         ("CAT-001", "CAT-002")),
    Item("FB-003", "DE",
         "Ich habe die blaue Variante bestellt und die graue bekommen. "
         "Das steht so nicht auf meiner Rechnung, Bestellnummer 55-204871.",
         ("CAT-003",)),
    Item("FB-004", "FR",
         "Le carton est arrive ecrase et il manque deux des quatre filtres. "
         "C'est la deuxieme fois ce mois-ci.",
         ("CAT-002", "CAT-004")),
    Item("FB-005", "NL",
         "De bezorger heeft het pakket bij de containers in de regen gezet "
         "en aangegeven dat er niemand thuis was. Ik stond bij de deur.",
         ("CAT-005",)),
    Item("FB-006", "UK",
         "Second time the screen arrived cracked. I have written about this "
         "packaging problem twice already and nobody has fixed it.",
         ("CAT-002", "CAT-006")),
    Item("FB-007", "ES",
         "Pedido 55-209004. Dijeron martes, llego el viernes, y nadie aviso. "
         "Ya van dos veces este mes.",
         ("CAT-001",)),
    Item("FB-008", "IT",
         "Ho ricevuto l'ordine di qualcun altro. Sulla fattura c'e un altro "
         "indirizzo. E' la prima volta che succede.",
         ("CAT-003",)),
    Item("FB-009", "SE",
         "Ladan kom utan stromadapter. Fakturan sager att den ingar. "
         "Ordernummer 55-211502.",
         ("CAT-004",)),
    Item("FB-010", "CZ",
         "Slibovali ctvrtek, prisilo v pondeli. Uz potreti posunuli datum "
         "a nikdo se neozval.",
         ("CAT-001",)),
    Item("FB-011", "US",
         "Great, another delivery date. That's the third one. Order 55-213880. "
         "At this point I just want to know where it actually is.",
         ("CAT-001",)),
    Item("FB-012", "JP",
         "The box was crushed on one corner and the unit inside is dented. "
         "This is the second time this month and nobody has fixed it.",
         ("CAT-002",)),
    Item("FB-013", "DE",
         "Vielen Dank, alles kam puenktlich und heil an. Nichts zu beanstanden.",
         ()),
    Item("FB-014", "PL",
         "Wszystko w porzadku, dostawa na czas, produkt dziala. Dziekuje.",
         ()),
    Item("FB-015", "UK",
         "Driver marked it delivered while I was standing at the door, then "
         "left it with a neighbour I never agreed to. Parcel also looks bashed.",
         ("CAT-002", "CAT-005")),
]

MARKETS = sorted({i.market for i in ITEMS})


def for_market(market: str):
    return [i for i in ITEMS if i.market == market]


def workload(repeats: int = 1):
    """A run: every item, `repeats` times. Mirrors a DEV batch, where the same
    corpus is re-processed after every prompt tweak."""
    return [i for _ in range(repeats) for i in ITEMS]


if __name__ == "__main__":
    print(f"items            {len(ITEMS)}")
    print(f"markets          {len(MARKETS)}  ({', '.join(MARKETS)})")
    multi = sum(1 for i in ITEMS if len(i.truth) > 1)
    none_ = sum(1 for i in ITEMS if not i.truth)
    print(f"multi-category   {multi}   (why the answer is a list)")
    print(f"no category      {none_}   (the answer may legitimately be empty)")
