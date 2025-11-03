"""
Numeric-aware RAG with Qdrant: smart query parser for numeric constraints.

What this script shows:
 1) Create a Qdrant collection with text vectors + numeric payloads
 2) Parse natural-language queries for numeric filters (>, <, between, etc.)
 3) Combine semantic search (embeddings) with structured numeric filtering
 4) (Optional) Handle "top/bottom N by <field>" sorting after retrieval

Dependencies:
    pip install qdrant-client sentence-transformers

Notes:
 - Replace the in-memory client with your persistent Qdrant endpoint.
 - Customize FIELD_ALIASES to your domain.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer

# ---------------------------
# 1) Config & helpers
# ---------------------------

# Canonical field names -> list of aliases that may appear in user queries
FIELD_ALIASES: Dict[str, List[str]] = {
    "gdp_per_capita": ["gdp per capita", "gdp/capita", "gdp pc", "gdp per person", "gdp"],
    "population": ["population", "pop"],
    "life_expectancy": ["life expectancy", "lifespan"],
}

# You can add categorical fields here if needed, but this example focuses on numeric ranges.

SUFFIX_MULTIPLIERS = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
    "t": 1_000_000_000_000,
}

@dataclass
class NumericConstraint:
    field: str
    gte: Optional[float] = None
    lte: Optional[float] = None
    gt: Optional[float] = None
    lt: Optional[float] = None

@dataclass
class RankDirective:
    direction: str  # 'top' or 'bottom'
    n: int
    field: str

# ---------------------------
# 2) Parsing utilities
# ---------------------------

def normalize_number(token: str) -> Optional[float]:
    """Convert '50k', '2.5M', '10,000', '$30', '30%' -> float (strip currency/percent).
    Returns None if not a number.
    """
    s = token.strip().lower()
    s = s.replace(",", "").replace("$", "").replace("%", "")
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([kmbt])?", s)
    if not m:
        try:
            return float(s)
        except ValueError:
            return None
    value = float(m.group(1))
    suf = m.group(2)
    if suf:
        value *= SUFFIX_MULTIPLIERS[suf]
    return value


def find_field_in_text(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Return list of (canonical_field, span) for each alias occurrence in text."""
    out = []
    lowered = text.lower()
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            for m in re.finditer(r"\b" + re.escape(alias) + r"\b", lowered):
                out.append((field, m.span()))
    # Sort by position in text, deterministic
    out.sort(key=lambda x: x[1][0])
    return out


def parse_rank_directive(query: str) -> Optional[RankDirective]:
    """Parse patterns like:
       - top 5 by gdp per capita
       - bottom 10 by population
    """
    q = query.lower()
    m = re.search(r"\b(top|bottom)\s+(\d+)\s+by\s+([a-z0-9\s/]+?)\b", q)
    if not m:
        return None
    direction, n_str, field_text = m.groups()
    n = int(n_str)
    # Map field_text to canonical field
    field_text = field_text.strip()
    for field, aliases in FIELD_ALIASES.items():
        for a in aliases:
            if field_text == a:
                return RankDirective(direction=direction, n=n, field=field)
    # fuzzy: if field_text contains alias
    for field, aliases in FIELD_ALIASES.items():
        for a in aliases:
            if a in field_text:
                return RankDirective(direction=direction, n=n, field=field)
    return None


def parse_numeric_constraints(query: str) -> List[NumericConstraint]:
    """Parse simple numeric constraints tied to fields.

    Supported constructs (order-insensitive, near-field matching):
      - gdp per capita >= 50k
      - population between 1m and 5m
      - life expectancy > 70
      - under 3k population
      - at least 20k gdp
      - gdp from 10k to 30k
      - no more than 1000 pop

    Strategy: detect field mentions and search in a window around each for numeric patterns.
    """
    constraints: Dict[str, NumericConstraint] = {}
    q = query.lower()
    field_spans = find_field_in_text(q)

    # If no explicit fields found, we cannot bind numbers confidently
    if not field_spans:
        return []

    def get_or_make(field: str) -> NumericConstraint:
        nc = constraints.get(field)
        if not nc:
            nc = NumericConstraint(field)
            constraints[field] = nc
        return nc

    WINDOW = 48  # chars around field mention to scan for numbers/operators

    # Operator phrases -> (gte, lte, gt, lt)
    OPS = [
        (r">=|at least|no less than|minimum of|\bmin\b", "gte"),
        (r"<=|at most|no more than|maximum of|\bmax\b", "lte"),
        (r">|over|above|more than|exceed(ing)?", "gt"),
        (r"<|under|below|less than|fewer than", "lt"),
    ]

    for field, (start, end) in field_spans:
        left = max(0, start - WINDOW)
        right = min(len(q), end + WINDOW)
        ctx = q[left:right]

        # 1) BETWEEN / FROM ... TO ...
        m = re.search(r"(between|from)\s+([$0-9.,kmbt%]+)\s+(and|to)\s+([$0-9.,kmbt%]+)", ctx)
        if m:
            v1 = normalize_number(m.group(2))
            v2 = normalize_number(m.group(4))
            if v1 is not None and v2 is not None:
                low, high = sorted([v1, v2])
                nc = get_or_make(field)
                nc.gte = low
                nc.lte = high

        # 2) Simple comparator near a number (e.g., "> 50k", "at least 20m")
        # Scan for number tokens and look behind/ahead for an operator phrase
        for num_m in re.finditer(r"[$0-9.,]+\s*[kmbt%]?", ctx):
            num_text = num_m.group(0)
            value = normalize_number(num_text)
            if value is None:
                continue
            # look behind 12 words, ahead 6 words for an operator phrase
            span_left = max(0, num_m.start() - 64)
            span_right = min(len(ctx), num_m.end() + 64)
            mini = ctx[span_left:span_right]
            for pat, key in OPS:
                if re.search(pat, mini):
                    nc = get_or_make(field)
                    setattr(nc, key, value)
                    break

        # 3) Loose pattern: "<number> (units) <field>" like "under 3m population" handled above.
        # If no op found but exact equals implied like "gdp 50000" -> treat as >= and <= same value
        if field not in constraints:
            # try pattern: field followed by a lone number
            m2 = re.search(r"\b([$0-9.,]+\s*[kmbt%]?)\b", ctx)
            if m2:
                value = normalize_number(m2.group(1))
                if value is not None:
                    nc = get_or_make(field)
                    nc.gte = value
                    nc.lte = value

    return list(constraints.values())


# ---------------------------
# 3) Demo data + Qdrant setup
# ---------------------------

def build_demo_collection(client: QdrantClient, model: SentenceTransformer, collection: str = "countries"):

    # check if collection already exists
    if client.collection_exists(collection):
        # delete it first (if you want to recreate it)
        client.delete_collection(collection)

    # now create the collection
    client.create_collection(
        collection_name = collection,
        vectors_config = VectorParams(size=model.get_sentence_embedding_dimension(), distance="Cosine")  # your vector configuration
    )

    # client.recreate_collection(
    #     collection_name=collection,
    #     vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance="Cosine"),
    # )

    docs = [
        {
            "id": 1,
            "name": "United States",
            "gdp_per_capita": 74000,
            "population": 331_000_000,
            "life_expectancy": 77.3,
            "desc": "The United States has a highly developed economy with significant innovation.",
        },
        {
            "id": 2,
            "name": "Japan",
            "gdp_per_capita": 42000,
            "population": 126_000_000,
            "life_expectancy": 84.6,
            "desc": "Japan is known for advanced technology, aging population, and high life expectancy.",
        },
        {
            "id": 3,
            "name": "India",
            "gdp_per_capita": 2700,
            "population": 1_380_000_000,
            "life_expectancy": 70.4,
            "desc": "India has a rapidly growing economy and very large population.",
        },
        {
            "id": 4,
            "name": "Norway",
            "gdp_per_capita": 89000,
            "population": 5_400_000,
            "life_expectancy": 83.2,
            "desc": "Norway has one of the highest GDP per capita values and high quality of life.",
        },
    ]

    points: List[PointStruct] = []
    for d in docs:
        vec = model.encode(d["desc"]).tolist()
        payload = {
            "name": d["name"],
            "desc": d["desc"],
            "gdp_per_capita": d["gdp_per_capita"],
            "population": d["population"],
            "life_expectancy": d["life_expectancy"],
        }
        points.append(PointStruct(id=d["id"], vector=vec, payload=payload))

    client.upsert(collection_name=collection, points=points)


# ---------------------------
# 4) Retrieval with numeric filters + optional ranking
# ---------------------------

def build_qdrant_filter(ncs: List[NumericConstraint]) -> Optional[Filter]:
    conds = []
    for nc in ncs:
        # Build a single Range that may include multiple bounds
        r = Range()
        if nc.gte is not None:
            r.gte = float(nc.gte)
        if nc.lte is not None:
            r.lte = float(nc.lte)
        if nc.gt is not None:
            r.gt = float(nc.gt)
        if nc.lt is not None:
            r.lt = float(nc.lt)
        conds.append(FieldCondition(key=nc.field, range=r))
    if not conds:
        return None
    return Filter(must=conds)


def search_with_numeric_awareness(
    client: QdrantClient,
    model: SentenceTransformer,
    collection: str,
    query: str,
    k: int = 8,
):
    # Parse constraints and rank directive
    constraints = parse_numeric_constraints(query)
    rank = parse_rank_directive(query)

    # Semantic vector
    qvec = model.encode(query).tolist()

    qfilter = build_qdrant_filter(constraints)

    res = client.query_points(
        collection_name=collection,
        with_vectors=qvec,
        limit=k,
        query_filter=qfilter,
    )

    # Optional: post-sort by rank directive
    if rank is not None:
        field = rank.field
        reverse = True if rank.direction == "top" else False
        res = sorted(
            res,
            key=lambda r: r.payload.get(field, float("nan")),
            reverse=reverse,
        )
        res = res[: rank.n]

    return res, constraints, rank


# ---------------------------
# 5) Example usage
# ---------------------------
if __name__ == "__main__":
    client = QdrantClient(":memory:")  # swap for host=..., port=... in real setups
    model = SentenceTransformer("all-MiniLM-L6-v2")

    build_demo_collection(client, model)

    example_queries = [
        "wealthy countries with gdp per capita >= 50k",
        "countries with population between 1m and 10m",
        "show top 2 by life expectancy",
        "high quality of life but under 10m population",
        "countries with gdp from 40k to 100k",
        "countries with gdp 89000",  # equals-ish fallback
    ]

    for q in example_queries:
        hits, ncs, rank = search_with_numeric_awareness(client, model, "countries", q, k=10)
        print("\nQuery:", q)
        print("Parsed constraints:", ncs)
        print("Rank directive:", rank)
        print("Results:")
        for h in hits.points:
            p = h.payload
            print(
                f" - {p['name']}: gdp_per_capita={p['gdp_per_capita']}, population={p['population']}, life_expectancy={p['life_expectancy']}"
            )