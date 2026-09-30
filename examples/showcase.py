#!/usr/bin/env python3
"""Warehouse order processing.

A small but realistic module used to showcase **python_beautifier**.  It models
a warehouse that accepts orders, reserves stock, prices the basket and ships
parcels.

The module is organised in four layers:

- *domain* types (:class:`OrderStatus`, :class:`LineItem`, :class:`Order`)
- an :class:`Inventory` that tracks stock levels
- pricing helpers such as :func:`price_basket`
- an :class:`OrderProcessor` that orchestrates everything

Example:
    >>> inv = Inventory({"ABC-0001": 10})
    >>> inv.reserve("ABC-0001", 3)
    True
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Protocol

import requests

from .config import Settings

__all__ = ["OrderStatus", "LineItem", "Order", "Inventory", "OrderProcessor", "price_basket"]
__version__ = "1.4.2"
__author__ = "Ada Lovelace"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAX_RATE = 0.2  # VAT applied to every taxable line
FREE_SHIPPING_FROM = 100.0
MAX_RETRIES: int = 3
SKU_PATTERN = re.compile(r"^[A-Z]{3}-\d{4}$")
CARRIERS = {"dhl": 1.5, "ups": 1.8, "post": 0.9}  # price per kilogram


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class OrderStatus(Enum):
    """Lifecycle of an order."""

    NEW = "new"  # just created, nothing reserved yet
    RESERVED = "reserved"  # stock is on hold
    PAID = "paid"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"

    @property
    def is_final(self) -> bool:
        """Whether no further transitions are possible."""
        return self in (OrderStatus.SHIPPED, OrderStatus.CANCELLED)


@dataclass
class LineItem:
    """One product line of an order.

    Attributes:
        sku: Stock keeping unit, e.g. ``ABC-1234``.
        quantity: Number of units ordered.
        price: Unit price in euros.
    """

    sku: str
    quantity: int = 1
    price: float = 0.0
    taxable: bool = True  # exempt items (books, food) set this to False
    tags: List[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        """Net price of the line."""
        return self.price * self.quantity


@dataclass
class Order:
    """A customer order.

    Attributes:
        id: Unique order number.
        items: The product lines.
        status: Current :class:`OrderStatus`.
    """

    id: int
    items: List[LineItem]
    status: OrderStatus = OrderStatus.NEW
    coupon: Optional[str] = None

    def add(self, item: LineItem) -> None:
        """Append *item*, merging it with an existing line for the same SKU."""
        for existing in self.items:
            if existing.sku == item.sku:
                existing.quantity += item.quantity
                break
        else:
            self.items.append(item)


class ShippingStrategy(Protocol):
    """Anything that can quote a shipping price."""

    def quote(self, weight_kg: float, country: str) -> float:
        """Return the shipping price for a parcel."""
        ...


class Carrier(ABC):
    """Base class for carriers that deliver parcels."""

    name: str = "generic"

    @abstractmethod
    def quote(self, weight_kg: float, country: str) -> float:
        """Price a parcel.

        Args:
            weight_kg: Parcel weight in kilograms.
            country: ISO country code of the destination.

        Returns:
            The price in euros.
        """

    def label(self) -> str:
        return f"{self.name.upper()} parcel"


class FlatRateCarrier(Carrier):
    """A carrier that charges per kilogram with a minimum fee."""

    def __init__(self, name: str, per_kg: float, minimum: float = 4.99):
        self.name = name
        self.per_kg = per_kg
        self.minimum = minimum

    def quote(self, weight_kg: float, country: str) -> float:
        price = max(self.minimum, weight_kg * self.per_kg)
        if country != "DE":
            price *= 1.75
        return round(price, 2)


# ---------------------------------------------------------------------------
# Stock keeping
# ---------------------------------------------------------------------------


class Inventory:
    """Tracks how many units of each SKU are on the shelf.

    Attributes:
        low_stock_threshold (int): Below this level a warning is logged.
    """

    low_stock_threshold = 5

    def __init__(self, stock: Optional[Dict[str, int]] = None, *, name: str = "main") -> None:
        """Create an inventory.

        Args:
            stock: Initial SKU to quantity mapping.
            name: Human readable warehouse name.
        """
        self.name = name
        self._stock: Dict[str, int] = dict(stock or {})
        self._reserved = defaultdict(int)
        self.history: List[str] = []

    @property
    def total_units(self) -> int:
        """Total number of units across all SKUs."""
        return sum(self._stock.values())

    def available(self, sku: str) -> int:
        """Units of *sku* that are not reserved yet."""
        return self._stock.get(sku, 0) - self._reserved[sku]

    def reserve(self, sku: str, quantity: int = 1) -> bool:
        """Try to reserve stock for an order.

        Args:
            sku (str): Stock keeping unit, e.g. ``"ABC-1234"``.
            quantity (int): How many units to reserve. Must be positive.

        Returns:
            bool: ``True`` when the stock was reserved, ``False`` if there was not enough.

        Raises:
            ValueError: If *quantity* is not positive or *sku* is malformed.
        """
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if not SKU_PATTERN.match(sku):
            raise ValueError(f"bad sku: {sku!r}")

        if self.available(sku) >= quantity:
            self._reserved[sku] += quantity
            self.history.append(f"reserved {quantity} x {sku}")
            return True
        else:
            log.warning("not enough %s (wanted %d)", sku, quantity)
            return False

    def release(self, sku: str, quantity: int) -> None:
        """Give reserved units back to the shelf."""
        held = self._reserved[sku]
        self._reserved[sku] = max(0, held - quantity)
        self.history.append(f"released {quantity} x {sku}")

    def restock(self, deliveries: Dict[str, int]) -> int:
        """Add delivered goods to the shelf.

        Args:
            deliveries: Mapping of SKU to delivered quantity.

        Returns:
            How many units were added in total.
        """
        added = 0
        for sku, qty in deliveries.items():
            if qty <= 0:
                continue  # nothing to add
            self._stock[sku] = self._stock.get(sku, 0) + qty
            added += qty
            if self._stock[sku] > 1000:
                log.info("%s is overstocked", sku)
        return added

    def low_stock(self) -> Iterator[str]:
        """Yield every SKU that is running low."""
        for sku in sorted(self._stock):
            if self.available(sku) < self.low_stock_threshold:
                yield sku

    @classmethod
    def from_file(cls, path: Path) -> "Inventory":
        """Load an inventory from a JSON file.

        Args:
            path: Location of the JSON document.

        Returns:
            A populated inventory. An empty one if the file is damaged.
        """
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except FileNotFoundError:
            log.warning("no inventory file at %s", path)
            return cls()
        except json.JSONDecodeError as exc:
            log.error("broken inventory file: %s", exc)
            return cls()
        else:
            log.info("loaded %d SKUs", len(data))
            return cls(data)
        finally:
            log.debug("inventory load finished")

    async def sync_remote(self, url: str, *, retries: int = MAX_RETRIES) -> bool:
        """Pull stock levels from the central warehouse.

        Args:
            url: Endpoint returning ``{"sku": quantity}`` JSON.
            retries: How many times to retry on network errors.

        Returns:
            ``True`` if the local stock was updated.
        """
        for attempt in range(1, retries + 1):
            try:
                await asyncio.sleep(0.1 * attempt)  # back off a little more each time
                response = await asyncio.to_thread(requests.get, url, timeout=5)
                response.raise_for_status()
            except requests.RequestException as exc:
                log.warning("sync attempt %d failed: %s", attempt, exc)
                continue
            self._stock.update(response.json())
            return True
        return False


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def price_basket(items, *, tier="standard", coupon=None, country="DE"):
    """Compute the final price of a basket.

    Parameters
    ----------
    items : Iterable[LineItem]
        The basket content.
    tier : {"standard", "silver", "gold"}, optional
        Customer loyalty tier. Gold customers get the best discount.
    coupon : str, optional
        A coupon code such as ``"SPRING10"``.
    country : str, default "DE"
        ISO country code used to look up the tax rate.

    Returns
    -------
    float
        Total price including tax and shipping.

    Raises
    ------
    ValueError
        If the basket is empty or the tier is unknown.
    """
    items = list(items)
    if not items:
        raise ValueError("empty basket")

    subtotal = sum(i.price * i.quantity for i in items)

    # Loyalty discount
    if tier == "gold":
        discount = 0.15
    elif tier == "silver":
        discount = 0.08
    elif tier == "standard":
        discount = 0.0
    else:
        raise ValueError(f"unknown tier {tier!r}")

    if coupon and coupon.startswith("SPRING"):
        discount += int(coupon[6:] or 5) / 100
    discount = min(discount, 0.5)  # never give away more than half

    taxable = sum(i.total for i in items if i.taxable)
    tax = taxable * (1 - discount) * (TAX_RATE if country == "DE" else 0.0)
    shipping = 0.0 if subtotal >= FREE_SHIPPING_FROM else 4.99

    # TODO: handle gift cards, they are not discountable
    # total = subtotal - gift_card
    total = subtotal * (1 - discount) + tax + shipping
    return round(total, 2)


def describe_event(event: dict) -> str:
    """Turn a raw warehouse event into a readable sentence.

    :param event: Decoded JSON event with a ``type`` key.
    :type event: dict
    :returns: A one line description.
    :rtype: str
    """
    match event:
        case {"type": "restock", "sku": sku, "qty": qty}:
            return f"{qty} units of {sku} arrived"
        case {"type": "order", "id": order_id}:
            return f"order #{order_id} placed"
        case {"type": "cancel", "id": order_id, "reason": str(reason)}:
            return f"order #{order_id} cancelled: {reason}"
        case {"type": kind}:
            return f"unknown event {kind!r}"
        case _:
            return "malformed event"


def parcel_weight(items: Iterable[LineItem], catalog: Dict[str, float]) -> float:
    """Sum up the weight of a parcel in kilograms.

    Args:
        items: Items in the parcel.
        catalog: SKU to unit weight (kg) lookup.

    Returns:
        The weight, rounded up to 100 g.
    """
    total = 0.0
    for item in items:
        weight = catalog.get(item.sku)
        if weight is None:
            log.warning("no weight for %s, assuming 1 kg", item.sku)
            weight = 1.0
        total += weight * item.quantity
    return math.ceil(total * 10) / 10


def find_cheapest_carrier(carriers: List[Carrier], weight: float, country: str) -> Optional[Carrier]:
    """Pick the carrier with the lowest quote.

    Args:
        carriers: Candidates to compare.
        weight: Parcel weight in kg.
        country: Destination country code.

    Returns:
        The winner, or ``None`` if nobody can deliver.
    """
    best, best_price = None, float("inf")
    for carrier in carriers:
        try:
            price = carrier.quote(weight, country)
        except NotImplementedError:
            continue
        if price < best_price:
            best, best_price = carrier, price
    return best


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class OrderProcessor:
    """Runs orders through reservation, payment and shipping."""

    def __init__(self, inventory: Inventory, settings: Optional[Settings] = None):
        self.inventory = inventory
        self.settings = settings
        self.carriers: List[Carrier] = [FlatRateCarrier(n, p) for n, p in CARRIERS.items()]

    def process(self, order: Order, *, dry_run: bool = False) -> OrderStatus:
        """Move *order* as far through the pipeline as possible.

        Args:
            order: The order to process.
            dry_run: When true nothing is reserved or shipped.

        Returns:
            The status the order ended up in.

        Raises:
            RuntimeError: If the order is already in a final state.
        """
        if order.status.is_final:
            raise RuntimeError(f"order {order.id} is already {order.status.value}")

        if not order.items:
            log.info("order %s is empty", order.id)
            return OrderStatus.CANCELLED

        reserved = []
        for item in order.items:
            if not self.inventory.reserve(item.sku, item.quantity):
                for sku, qty in reserved:
                    self.inventory.release(sku, qty)
                order.status = OrderStatus.CANCELLED
                return order.status
            reserved.append((item.sku, item.quantity))

        order.status = OrderStatus.RESERVED
        if dry_run:
            for sku, qty in reserved:
                self.inventory.release(sku, qty)
            return order.status

        total = price_basket(order.items, coupon=order.coupon)
        with self._payment_session() as session:
            if session.charge(order.id, total):
                order.status = OrderStatus.PAID
            else:
                order.status = OrderStatus.CANCELLED

        if order.status is OrderStatus.PAID:
            self.ship(order)
        return order.status

    def ship(self, order: Order) -> None:
        """Choose a carrier and hand the parcel over."""
        weight = parcel_weight(order.items, {})
        carrier = find_cheapest_carrier(self.carriers, weight, "DE")
        if carrier is None:
            raise RuntimeError("no carrier available")
        log.info("shipping order %s with %s", order.id, carrier.label())
        order.status = OrderStatus.SHIPPED

    def _payment_session(self):
        raise NotImplementedError("wire up your payment provider")

    def retry(self, attempts: int = MAX_RETRIES):
        """Build a decorator that retries a flaky call.

        Args:
            attempts: Maximum number of tries.

        Returns:
            A decorator.
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                last_error = None
                for n in range(attempts):
                    try:
                        return func(*args, **kwargs)
                    except (OSError, TimeoutError) as err:
                        last_error = err
                        log.warning("attempt %d/%d failed", n + 1, attempts)
                raise RuntimeError("gave up") from last_error

            return wrapper

        return decorator


def load_catalog(path: str = "catalog.txt") -> Dict[str, float]:
    """Read ``SKU weight`` pairs from a text file.

    Args:
        path: File with one ``SKU weight`` pair per line.

    Returns:
        Mapping of SKU to weight in kilograms.
    """
    catalog: Dict[str, float] = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sku, _, weight = line.partition(" ")
            try:
                catalog[sku] = float(weight)
            except ValueError:
                log.error("line %d: bad weight %r", number, weight)
    return catalog


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    inventory = Inventory({"ABC-0001": 10, "XYZ-0002": 3})
    processor = OrderProcessor(inventory)
    demo = Order(1, [LineItem("ABC-0001", 2, 19.9)])
    print(processor.process(demo, dry_run=True))
