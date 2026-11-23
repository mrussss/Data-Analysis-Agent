"""Business benchmark inputs and scorer-only expectations.

Olist tasks are observational. Retail tasks use an independent fictional panel;
the event catalog contains public-facing records, with no evaluation labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import duckdb

from .business_tools import BusinessGoal


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    goal: BusinessGoal
    verdict: str
    findings: tuple[str, ...]
    events: tuple[tuple[str, str, int, str, str], ...] = ()
    # Synthetic interventions are scorer/build inputs only.
    changes: tuple[tuple[str, int, str, str, int | None], ...] = ()


def historical(id, kind, question, first, second, required, *, category=None, metric="item_lines", dimension=None):
    def month(value):
        start = date.fromisoformat(f"2018-{value:02}-01")
        return start, date.fromisoformat(f"2018-{value + 1:02}-01")

    return Case(id, kind, BusinessGoal(question, "olist", month(first), month(second),
                                      category, metric, dimension),
                "statistical" if kind == "A" else "association", tuple(required.split()))


H = historical
OLIST_TEST = (
    H("weekday_offsets", "A", "Which weekday contributed the largest absolute order-count change? Identify its direction.",
      4, 5, "weekday:top=7:down", metric="orders", dimension="weekday"),
    H("state_gmv", "A", "Which buyer state contributed the largest absolute item GMV change?",
      7, 8, "state:top=SP:up", metric="gmv", dimension="state"),
    H("category_gmv", "A", "Which product category contributed the largest absolute GMV change?",
      5, 6, "category:top=watches_gifts:down", metric="gmv", dimension="category"),
    H("seller_geography", "A", "Which seller state contributed the largest absolute change in computer accessory item lines?",
      3, 4, "seller_state:top=SP:down", category="computers_accessories", dimension="seller_state"),
    H("status_orders", "A", "Which order status contributed the largest absolute change in order count?",
      2, 3, "status:top=delivered:up", metric="orders", dimension="status"),
    H("weekday_freight", "A", "Which purchase weekday contributed the largest absolute change in housewares freight charged?",
      5, 6, "weekday:top=2:down", category="housewares", metric="freight_value", dimension="weekday"),
    H("delivery_mix", "A", "Did the late-delivery rate fall mostly through buyer-state composition or within-state rates? Identify the biggest state contribution.",
      3, 4, "state:effect=within state:top=SP:down", metric="late_delivery_rate", dimension="state"),
    H("offsetting_revenue", "A", "Which telephony seller state contributed the largest absolute GMV change?",
      6, 7, "seller_state:top=SP:down", category="telephony", metric="gmv", dimension="seller_state"),
    H("volume_value_diverge", "B", "Telephony GMV rose. Compare the material changes in line volume and unit value and check whether fulfillment worsened.",
      2, 3, "headline:item_lines=stable headline:gmv=up headline:avg_price=up delivery:late_rate=up",
      category="telephony"),
    H("reviews_despite_delivery", "B", "Housewares complaints reportedly rose. Did fulfillment worsen too? Compare both rates and check order review coverage.",
      5, 6, "reviews:negative_rate=up reviews:coverage=stable delivery:late_rate=down", category="housewares"),
    H("lower_volume_better_experience", "B", "Computer accessories lost volume. Did customers fare worse, or did both review and delivery outcomes improve?",
      3, 4, "headline:item_lines=down reviews:negative_rate=down delivery:late_rate=down",
      category="computers_accessories"),
    H("gross_vs_freight", "B", "Housewares GMV and freight may be moving in different directions. Compare item prices, purchase volume and freight charged per line.",
      7, 8, "headline:item_lines=up headline:avg_price=down freight:freight_per_line=down", category="housewares"),
    H("price_revenue_offset", "B", "Cool stuff GMV plunged. Compare line volume and implied item value, then test whether breadth of products sold also plunged.",
      7, 8, "headline:gmv=down headline:item_lines=stable headline:avg_price=down products:active_products=up",
      category="cool_stuff"),
    H("seller_breadth", "B", "Office furniture item lines fell. Did seller breadth contract too? Check whether delivery also deteriorated.",
      7, 8, "headline:item_lines=down sellers:active_sellers=down delivery:late_rate=stable", category="office_furniture"),
    H("basket_vs_orders", "B", "For bed and bath, did the July item-line fall come with fewer orders or materially fewer category lines per order?",
      6, 7, "headline:item_lines=down headline:category_orders=down headline:lines_per_order=stable",
      category="bed_bath_table"),
    H("ratings_by_delivery", "B", "For health and beauty, compare the change in complaints with late delivery and check whether late and on-time orders had different complaint rates.",
      3, 4, "reviews:negative_rate=down delivery:late_rate=down review_delivery:late_gap=positive",
      category="health_beauty"),
)

ANCHOR = date(2025, 6, 15)
DAYS = timedelta(days=14)


def retail(id, kind, entity, question, verdict, required, events=(), changes=()):
    return Case(id, kind, BusinessGoal(question, "retail", (ANCHOR - DAYS, ANCHOR),
                                      (ANCHOR, ANCHOR + DAYS), entity, "units"),
                verdict, tuple(required.split()), events, changes)


R = retail
RETAIL_TEST = (
    R("recall_scope", "C", "Fable Circuit", "Investigate the unit decline and complaint increase at Fable Circuit. Determine which explanation has evidence, considering public records and operational alternatives.",
      "plausible", "headline:units=down scope:sku=core complaints:complaint_rate=up stock_core:availability=stable events:recall=matched",
      (("recall", "Fable Circuit", -1, "core", "The fictional safety office lists a voluntary recall for the core model."),),
      (("units", 0, "core", "*", 22), ("complaints", 0, "core", "*", 5))),
    R("voucher_expiry", "C", "Morrow Loom", "Investigate the sales drop at Morrow Loom. Check whether public commercial activity and paid prices line up.",
      "plausible", "headline:units=down pricing:price=up pricing:promo_rate=down events:voucher=matched",
      (("voucher", "Morrow Loom", -1, "all", "Merchant bulletin. The summer checkout voucher ends tonight."),),
      (("price", -14, "*", "*", 75), ("promo", -14, "*", "*", 1),
       ("price", 0, "*", "*", 100), ("promo", 0, "*", "*", 0), ("units", 0, "*", "*", 31))),
    R("distribution_region", "C", "Kite Harbor", "Kite Harbor lost sales. Identify the affected region and check channel access and public records.",
      "plausible", "headline:units=down scope:region=north access:visits=down events:channel=matched",
      (("channel", "Kite Harbor", -2, "north", "A northern shopping portal announces a temporary category listing pause."),),
      (("units", 0, "*", "north", 18), ("visits", 0, "*", "north", 35))),
    R("market_closure", "C", "Dusk Sprig", "Dusk Sprig lost sales. Determine whether the peer market moved too and whether the timing fits a public calendar event.",
      "plausible", "headline:units=down peer:units=down timeline:onset=2025-06-19 events:closure=matched",
      (("closure", "Dusk Sprig", 4, "market", "A market calendar lists a retail trading recess starting today; the listing includes this merchant."),),
      (("units", 4, "*", "*", 12),)),
    R("competitor_launch", "C", "Vela Orchard", "Vela Orchard units fell. Check whether peer units rose and whether an external market entry fits the timing. Assess whether an internal price change is supported.",
      "plausible", "headline:units=down peer:units=up pricing:price=stable events:entry=matched",
      (("entry", "Vela Orchard", -1, "market", "A fictional trade circular lists a competing line in this merchant’s directly comparable category."),),
      (("units", 0, "*", "*", 30),)),
    R("resupply", "C", "Pollen Arc", "Pollen Arc units recovered. Check whether an inventory change and a distributor update line up with the recovery.",
      "plausible", "headline:units=up stock:availability=up events:resupply=matched",
      (("resupply", "Pollen Arc", 0, "all", "A fictional distributor records the reopening of the replenishment lane."),),
      (("units", -14, "*", "*", 0), ("available", -14, "*", "*", 0),
       ("units", 0, "*", "*", 50), ("available", 0, "*", "*", 1))),
    R("visits_without_conversion", "C", "Grove Relay", "After an apparent visibility boost at Grove Relay, did traffic or sales actually improve? Check whether conversion kept pace and find the relevant public record.",
      "plausible", "headline:units=stable access:visits=up access:conversion=down events:feature=matched",
      (("feature", "Grove Relay", -1, "all", "A shopping guide announces a homepage feature for this merchant."),),
      (("visits", 0, "*", "*", 170),)),
    R("complaints_without_fulfillment", "C", "Copper Finch", "Copper Finch saw a complaint increase. Check whether delivery deteriorated and look for a product-specific public update.",
      "plausible", "complaints:complaint_rate=up service:late_rate=stable scope:complaints_sku=plus events:firmware=matched",
      (("firmware", "Copper Finch", -1, "plus", "Release notes describe an interface change on the plus model."),),
      (("complaints", 0, "plus", "*", 6),)),
    R("stock_vs_bad_press", "D", "Flint Weave", "Flint Weave units fell. Weigh a possible public reputational issue against direct operational constraints.",
      "observed_constraint", "headline:units=down stock:availability=down stock:stockout_sales_loss=present events:press=matched",
      (("press", "Flint Weave", -1, "all", "A fictional newspaper covers criticism of the merchant's spokesperson."),),
      (("units", 0, "core", "*", 0), ("available", 0, "core", "*", 0))),
    R("wrong_entity", "D", "Sable Loom", "Sable Loom lost sales. A similarly named merchant may be in the news. Check whether the subject matches and whether there is a competing stock signal.",
      "insufficient", "headline:units=down events:other=wrong_subject stock:availability=stable",
      (("other", "Sable Loom Outlet", -1, "all", "A fictional product bulletin lists a legal complaint about Sable Loom Outlet."),),
      (("units", 0, "*", "*", 33),)),
    R("after_onset", "D", "Iris Forge", "Iris Forge lost sales. Could the public incident explain the onset? Establish the first sustained break.",
      "insufficient", "headline:units=down timeline:onset=2025-06-17 events:incident=after",
      (("incident", "Iris Forge", 6, "all", "A fictional press release reports a spokesperson controversy."),),
      (("units", 2, "*", "*", 34),)),
    R("two_events", "D", "Ashen Vale", "Ashen Vale units fell. Assess whether the simultaneous public events can be distinguished with the available internal records.",
      "ambiguous", "headline:units=down events:article=matched events:policy=matched stock:availability=stable",
      (("article", "Ashen Vale", 0, "all", "A fictional trade paper reports negative publicity."),
       ("policy", "Ashen Vale", 0, "all", "A fictional marketplace announces a change to category search ranking.")),
      (("units", 0, "*", "*", 34),)),
    R("stock_unobserved", "D", "Quill Beacon", "Quill Beacon lost sales. Can inventory or a public event explain the drop? Check inventory coverage directly.",
      "insufficient", "headline:units=down stock:availability=missing events:none=found",
      (), (("units", 0, "*", "*", 34), ("available", -14, "*", "*", None))),
    R("no_movement", "D", "Rook Linen", "A negative item appeared about Rook Linen. Determine if there was any material change in sales or complaints.",
      "stable", "headline:units=stable complaints:complaint_rate=stable events:story=matched",
      (("story", "Rook Linen", 0, "all", "A fictional columnist criticizes the supplier’s campaign."),)),
)

DEV = (
    H("dev_status", "A", "Which order status contributed the largest absolute change in order count?",
      5, 6, "status:top=delivered:down", metric="orders", dimension="status"),
    H("dev_experience", "B", "Did auto late delivery and bad reviews improve together? Check the two metrics.",
      3, 4, "delivery:late_rate=down reviews:negative_rate=down", category="auto"),
    R("dev_campaign", "C", "Wren Parcel", "Wren Parcel unit growth coincided with a possible public special. Check paid prices and promotion activity.",
      "plausible", "headline:units=up pricing:price=down pricing:promo_rate=up events:offer=matched",
      (("offer", "Wren Parcel", 0, "all", "A fictional storefront announces a limited checkout offer."),),
      (("price", 0, "*", "*", 78), ("units", 0, "*", "*", 70), ("promo", 0, "*", "*", 1))),
    R("dev_wrong_timing", "D", "Linden Cove", "Could a reported incident account for the onset of the unit change at Linden Cove? Check the daily series.",
      "insufficient", "headline:units=down timeline:onset=2025-06-15 events:notice=after",
      (("notice", "Linden Cove", 8, "all", "A fictional trade report describes a public dispute."),),
      (("units", 0, "*", "*", 34),)),
)
TEST = OLIST_TEST + RETAIL_TEST
ONLINE_PAIRS = ("category_gmv", "ratings_by_delivery", "recall_scope", "after_onset", "stock_vs_bad_press")


def build_retail(case: Case, target):
    """Create a per-case panel and return only the public event records."""
    rows = []
    for day in range(-14, 14):
        when = ANCHOR + timedelta(days=day)
        for region in ("north", "south"):
            for sku in ("core", "plus"):
                values = dict(units=50, price=100, promo=0, available=1, visits=100, complaints=1, late=2)
                for field, start, item, area, value in case.changes:
                    if day >= start and item in ("*", sku) and area in ("*", region):
                        values[field] = value
                if not values["units"]:
                    values.update(complaints=0, late=0)
                rows.append((when, case.goal.entity, sku, region, *values.values()))
        for region in ("north", "south"):
            peer = {"market_closure": 12 if day >= 4 else 50,
                    "competitor_launch": 72 if day >= 0 else 50}.get(case.id, 50)
            rows.append((when, "market peer", "peer", region, peer, 100, 0, 1, 100, 1, 2))
    with duckdb.connect(str(target)) as db:
        db.execute("""CREATE TABLE retail (day DATE, entity VARCHAR, sku VARCHAR, region VARCHAR,
                    units INT, price INT, promo INT, available INT, visits INT, complaints INT, late INT)""")
        db.executemany("INSERT INTO retail VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    records = tuple({"id": id, "subject": subject, "date": (ANCHOR + timedelta(days=offset)).isoformat(),
                     "scope": scope, "text": text, "source": "fictional benchmark archive"}
                    for id, subject, offset, scope, text in case.events)
    return records
