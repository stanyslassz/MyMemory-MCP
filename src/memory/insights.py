"""ACT-R cognitive insights from graph data."""

from datetime import date

from src.core.models import GraphData


def compute_insights(graph: GraphData, today_str: str = None) -> dict:
    """Analyze graph for ACT-R-aware insights. Zero LLM."""
    today = date.fromisoformat(today_str) if today_str else date.today()

    insights = {
        "total_entities": len(graph.entities),
        "total_relations": len(graph.relations),
        "forgetting_curve": [],
        "emotional_hotspots": [],
        "weak_relations": [],
        "network_hubs": [],
        "scoring_distribution": {
            "0.0-0.1": 0,
            "0.1-0.3": 0,
            "0.3-0.5": 0,
            "0.5-0.7": 0,
            "0.7-1.0": 0,
        },
    }

    for eid, e in graph.entities.items():
        s = e.score
        if s < 0.1:
            insights["scoring_distribution"]["0.0-0.1"] += 1
        elif s < 0.3:
            insights["scoring_distribution"]["0.1-0.3"] += 1
        elif s < 0.5:
            insights["scoring_distribution"]["0.3-0.5"] += 1
        elif s < 0.7:
            insights["scoring_distribution"]["0.5-0.7"] += 1
        else:
            insights["scoring_distribution"]["0.7-1.0"] += 1

        # Entities near forgetting threshold (score > 0 but <= 0.1)
        if 0.0 < s <= 0.1:
            insights["forgetting_curve"].append({
                "entity": eid,
                "title": e.title,
                "score": round(s, 4),
            })

        # Emotional hotspots (high negative valence ratio)
        nvr = getattr(e, "negative_valence_ratio", 0.0) or 0.0
        if nvr > 0.3:
            insights["emotional_hotspots"].append({
                "entity": eid,
                "title": e.title,
                "ratio": round(nvr, 3),
            })

    # Weak relations
    for r in graph.relations:
        if r.strength < 0.2:
            age = 0
            if r.last_reinforced:
                try:
                    age = (today - date.fromisoformat(str(r.last_reinforced))).days
                except (ValueError, TypeError):
                    pass
            insights["weak_relations"].append({
                "from": r.from_entity,
                "to": r.to_entity,
                "type": r.type,
                "strength": round(r.strength, 3),
                "days_since_reinforced": age,
            })

    # Network hubs (top 10 by degree)
    degree: dict[str, int] = {}
    for r in graph.relations:
        degree[r.from_entity] = degree.get(r.from_entity, 0) + 1
        degree[r.to_entity] = degree.get(r.to_entity, 0) + 1
    top_hubs = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
    for eid, d in top_hubs:
        e = graph.entities.get(eid)
        insights["network_hubs"].append({
            "entity": eid,
            "title": e.title if e else eid,
            "degree": d,
        })

    return insights
