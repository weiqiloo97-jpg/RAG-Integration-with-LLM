import json

versions = [
    ("v5.2.3", "twbs_bootstrap", "Bootstrap"),
    ("v5.3.1", "twbs_bootstrap", "Bootstrap"),
    ("v5.3.2", "twbs_bootstrap", "Bootstrap"),
    ("v5.3.3", "twbs_bootstrap", "Bootstrap"),
    ("v5.3.4", "twbs_bootstrap", "Bootstrap"),
    ("v5.3.5", "twbs_bootstrap", "Bootstrap"),
    ("v2.4.7", "Apache Spark Release Notes", "Spark"),
    ("v3.3.4", "Apache Spark Release Notes", "Spark"),
    ("v3.4.4", "Apache Spark Release Notes", "Spark"),
    ("v3.5.3", "Apache Spark Release Notes", "Spark"),
    ("v3.5.4", "Apache Spark Release Notes", "Spark"),
    ("v3.5.5", "Apache Spark Release Notes", "Spark"),
]

topics_bootstrap = [
    "color mode improvements", "CSS utility classes", "Sass variable defaults",
    "JavaScript selector engine", "modal backdrop focus", "dropdown positioning",
    "tooltip boundary alignment", "navbar dark mode", "popover arrow offsets",
    "flexbox grid spacing", "form control validation styling", "button component contrast",
    "badge padding defaults", "carousel transition timers", "accordion item collapse"
]

topics_spark = [
    "memory management optimizations", "structured streaming watermark fixes",
    "PySpark DataFrame API methods", "SQL query planner rules", "K8s executor pod scaling",
    "Parquet reader column pruning", "Orc file format handling", "ANSI SQL compatibility mode",
    "delta lake protocol sync", "spark submit script flags"
]

queries = []
q_counter = 1

for ver, doc_name, doc_type in versions:
    topics = topics_bootstrap if doc_type == "Bootstrap" else topics_spark
    for t in topics:
        if q_counter > 100:
            break
        q_text = f"What {t} were introduced or modified in {doc_type} {ver}?"
        queries.append({
            "query_id": f"q{q_counter}",
            "query": q_text,
            "expected_version": ver,
            "source_document": doc_name,
            "target_version": ver
        })
        q_counter += 1

# Pad to exactly 100 queries if needed
while len(queries) < 100:
    ver, doc_name, doc_type = versions[len(queries) % len(versions)]
    q_text = f"List the release notes and bug fixes for {doc_type} release {ver} item #{len(queries)+1}."
    queries.append({
        "query_id": f"q{len(queries)+1}",
        "query": q_text,
        "expected_version": ver,
        "source_document": doc_name,
        "target_version": ver
    })

with open("RAG_evaluation/test_queries_100.json", "w", encoding="utf-8") as f:
    json.dump(queries[:100], f, indent=2)

print(f"Generated {len(queries[:100])} queries in RAG_evaluation/test_queries_100.json")
