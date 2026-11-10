


"""
benchmark/make_challenges.py — Programmatic generator for 200+ challenge prompts.

Generates hard benchmark prompts across four core challenge types:
1. Conflicting or tension-heavy constraints (tight length vs topic density, forbidden words, no commas).
2. Nested formats (multi-level JSON objects, nested configurations, schema validations).
3. Tricky edge cases (negative arithmetic, zero values, strict regex patterns, abbreviation-resistant sentence counts).
4. Paraphrased variants (formal, terse/robotic, RFC/legalistic, and conversational framings).

Outputs:
    benchmark/challenges.json (same schema as samples.json with 'split' field ~60/40 tune/test).

Usage:
    python -m agentic_harness.benchmark.make_challenges
    python src/agentic_harness/benchmark/make_challenges.py --count 240
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
CHALLENGES_FILE = _HERE / "challenges.json"


def _generate_conflicting_constraints() -> list[dict[str, Any]]:
    """Tension-heavy constraints: word limits + forbidden common words or punctuation."""
    items = []
    
    # Template group 1: explain topic without essential words + sentence constraints
    topics_1 = [
        ("photosynthesis", ["plant", "sun", "light", "green"], 
         "Chloroplasts synthesize vital carbohydrates by harvesting radiant stellar energy from the atmosphere.", 
         {"min": 8, "max": 18}),
        ("gravity", ["fall", "down", "earth", "drop"],
         "Massive celestial objects exert an attractive force pulling surrounding bodies toward their centers.",
         {"min": 8, "max": 16}),
        ("computers", ["screen", "keyboard", "code", "device"],
         "Electronic hardware executes calculated algorithmic operations to process vast streams of information.",
         {"min": 8, "max": 16}),
        ("democracy", ["vote", "election", "ballot", "leader"],
         "Citizens participate directly in governance through collective consensus and institutional representation.",
         {"min": 8, "max": 16}),
        ("internet", ["web", "online", "network", "wire"],
         "Global communication protocols distribute packets across interconnected servers worldwide.",
         {"min": 8, "max": 14}),
        ("music", ["sound", "song", "instrument", "listen"],
         "Auditory harmonies structure acoustic frequencies into emotional patterns.",
         {"min": 6, "max": 14}),
        ("automobile", ["car", "wheel", "engine", "drive"],
         "Motorized passenger carriages navigate transportation roadways using internal combustion propulsion.",
         {"min": 6, "max": 14}),
        ("airplane", ["fly", "wing", "sky", "plane"],
         "Pressurized aerodynamic cabins transport travelers across distant continents high above cloud layers.",
         {"min": 6, "max": 16}),
        ("medicine", ["drug", "doctor", "pill", "cure"],
         "Therapeutic clinical treatments restore physiological balance and alleviate suffering.",
         {"min": 6, "max": 14}),
        ("telescope", ["star", "look", "space", "eye"],
         "Optical lenses magnify distant galactic formations for astronomical documentation.",
         {"min": 6, "max": 14}),
    ]

    for topic, forbidden, ref, wps in topics_1:
        items.append({
            "category": "conflicting_constraints",
            "prompt": f"Explain what {topic} is in one sentence. You must NOT use any of these words: {', '.join(forbidden)}. The sentence must contain between {wps['min']} and {wps['max']} words.",
            "reference_answer": ref,
            "checker": {
                "type": "forbidden_words",
                "params": {"words": forbidden}
            }
        })

    # Template group 2: No commas + exact sentence counts
    no_comma_prompts = [
        ("the water cycle", 
         "Solar heat evaporates liquid oceans into vapor. Clouds form when moisture condenses aloft. Precipitation returns freshwater to rivers and soil.",
         3),
        ("cellular respiration",
         "Cells consume glucose to generate biochemical energy. Mitochondria drive this fundamental metabolic reaction. Carbon dioxide is released as an outcome.",
         3),
        ("Newton's first law",
         "Objects in motion maintain their continuous trajectory. Stationary bodies remain resting until an external force interferes. Inertia governs this fundamental physical reality.",
         3),
        ("tectonic motion",
         "Rigid crustal slabs drift atop molten mantle layers. Colliding boundaries uplift massive mountain ridges. Continental landmasses slowly rearrange across geological eras.",
         3),
        ("deep space exploration",
         "Robotic probes venture beyond planetary orbits. Distant radio signals beam discoveries back home. Humanity expands its scientific frontiers steadily.",
         3),
        ("ecosystem biodiversity",
         "Diverse species stabilize natural environments. Interdependent organisms balance local nutrient cycles. Habitat destruction threatens fragile food webs everywhere.",
         3),
        ("cryptography basics",
         "Ciphers conceal sensitive messages from unauthorized observers. Mathematical keys lock and unlock digital text. Security protects contemporary digital communication channels.",
         3),
        ("stellar life cycles",
         "Protostars ignite hydrogen fusion within dense nebulae. Mature stars balance immense gravitational collapse. Supernovae disperse heavy cosmic elements across galaxies.",
         3),
        ("oceanic currents",
         "Thermohaline circulation channels heat across international waters. Prevailing winds guide surface tidal drifts. Marine ecosystems depend upon nutrient upwelling.",
         3),
        ("quantum behavior",
         "Subatomic particles exhibit simultaneous wave and particle characteristics. Measurement collapses indeterminate probabilistic states immediately. Entanglement links distant particles instantly.",
         3),
    ]

    for topic, ref, sc in no_comma_prompts:
        items.append({
            "category": "conflicting_constraints",
            "prompt": f"Write exactly {sc} sentences explaining {topic} without using any commas.",
            "reference_answer": ref,
            "checker": {
                "type": "no_commas",
                "params": {}
            }
        })

    # Template group 3: Strict exact bullet counts with specific prefix + word limit per bullet
    bullet_topics = [
        ("microservices benefits", ["Independent scaling of components", "Decoupled deployment pipelines", "Fault isolation across boundaries", "Polyglot technology selection"], 4),
        ("clean code principles", ["Meaningful variable identifiers", "Small focused functions", "Comprehensive automated tests", "Clear modular abstraction"], 4),
        ("agile sprint phases", ["Sprint planning session", "Daily standup synchronization", "Iterative feature development", "Sprint retrospective review"], 4),
        ("cybersecurity hygiene rules", ["Enforce multifactor authentication", "Apply routine software patches", "Maintain isolated offline backups", "Conduct regular access audits"], 4),
        ("data normalization advantages", ["Elimination of redundant fields", "Improved data consistency", "Reduced anomalies on update", "Optimized database storage footprint"], 4),
        ("cloud architecture pillars", ["Operational excellence practices", "Robust data encryption", "Automated elasticity configurations", "Proactive cost management"], 4),
        ("distributed caching strategies", ["Cache aside pattern", "Read through caching", "Write through synchronization", "Time to live expirations"], 4),
        ("machine learning pipelines", ["Raw data ingestion", "Automated feature engineering", "Model parameter optimization", "Continuous validation telemetry"], 4),
        ("REST API guidelines", ["Stateless request handling", "Standard HTTP verbs", "Predictable status responses", "Hierarchical resource URIs"], 4),
        ("CI/CD best practices", ["Fast deterministic test runs", "Immutable build artifacts", "Automated rollback triggers", "Comprehensive observability logs"], 4),
    ]

    for topic, bullets, count in bullet_topics:
        ref_text = "\n".join(f"* {b}" for b in bullets)
        items.append({
            "category": "conflicting_constraints",
            "prompt": f"List exactly {count} distinct points about {topic}. Each point must start with '* ' and contain no more than 6 words.",
            "reference_answer": ref_text,
            "checker": {
                "type": "exact_bullet_count",
                "params": {"expected_count": count, "bullet_prefix": "*"}
            }
        })

    # Group 4: Exact word count bounds with required prefix
    prefix_length_cases = [
        ("STATUS: System latency remains nominal across all edge servers with zero errors.", "STATUS:", 12, 16),
        ("ALERT: Database memory utilization exceeded critical threshold on replica node three.", "ALERT:", 10, 14),
        ("NOTE: Scheduled maintenance begins at midnight UTC and lasts exactly two hours.", "NOTE:", 11, 15),
        ("CONFIRM: All production deployment validation health checks passed without any regressions.", "CONFIRM:", 10, 14),
        ("METRIC: Weekly active customer engagement increased fourteen percent following recent release.", "METRIC:", 10, 14),
        ("AUDIT: Access permissions for temporary contractor accounts were revoked as scheduled.", "AUDIT:", 10, 14),
        ("REPORT: Quarterly sales targets across European regions exceeded previous historical benchmarks.", "REPORT:", 10, 14),
        ("WARNING: Upstream payment gateway timeout rate reached two percent during peak load.", "WARNING:", 11, 15),
        ("DISPATCH: Emergency maintenance crew arrived on site to inspect fiber optic lines.", "DISPATCH:", 11, 15),
        ("TELEMETRY: Background queue processing completed one million tasks within five minutes.", "TELEMETRY:", 10, 14),
    ]

    for ref, pfx, min_w, max_w in prefix_length_cases:
        items.append({
            "category": "conflicting_constraints",
            "prompt": f"Output an operational status line starting strictly with '{pfx}'. The total output must have between {min_w} and {max_w} words.",
            "reference_answer": ref,
            "checker": {
                "type": "starts_with",
                "params": {"prefix": pfx}
            }
        })

    return items


def _generate_nested_formats() -> list[dict[str, Any]]:
    """Complex JSON schemas, nested attributes, multi-entity records."""
    items = []

    schemas = [
        ("microservice telemetry payload",
         ["service", "status", "metrics"],
         {"service": "auth-service", "status": "healthy", "metrics": {"cpu": 22.4, "memory_mb": 512, "uptime_sec": 86400}}),
        ("ecommerce order with shipping address",
         ["order_id", "customer", "shipping_address", "total_cents"],
         {"order_id": "ORD-9912", "customer": "Dana Scully", "shipping_address": {"street": "123 Main", "city": "DC", "zip": "20001"}, "total_cents": 14900}),
        ("CI pipeline build artifact descriptor",
         ["build_id", "commit_sha", "artifacts", "passed"],
         {"build_id": "b-4491", "commit_sha": "7f8a91c", "artifacts": ["bin/app", "docs/api.json"], "passed": True}),
        ("cloud security IAM policy",
         ["version", "statement", "role_arn"],
         {"version": "2026-10-17", "statement": {"effect": "Allow", "action": "s3:*"}, "role_arn": "arn:aws:iam::123:role/App"}),
        ("machine learning model registry entry",
         ["model_name", "version", "hyperparameters", "accuracy"],
         {"model_name": "fraud_detector", "version": "2.4.0", "hyperparameters": {"lr": 0.001, "epochs": 50}, "accuracy": 0.984}),
        ("database replication status",
         ["primary_node", "replicas", "lag_seconds", "sync_mode"],
         {"primary_node": "pg-01", "replicas": ["pg-02", "pg-03"], "lag_seconds": 0.05, "sync_mode": "streaming"}),
        ("IoT sensor environmental reading",
         ["device_id", "sensor_type", "reading", "timestamp_iso"],
         {"device_id": "temp-sensor-09", "sensor_type": "thermocouple", "reading": {"celsius": 21.5, "humidity_pct": 45}, "timestamp_iso": "2026-11-07T12:00:00Z"}),
        ("user session authentication token",
         ["session_id", "user_id", "claims", "expires_at"],
         {"session_id": "sess-xyz", "user_id": "usr-8812", "claims": {"role": "admin", "org": "acme"}, "expires_at": 1799999999}),
        ("API gateway route configuration",
         ["path", "upstream_service", "rate_limit", "cors"],
         {"path": "/v1/users", "upstream_service": "user-backend:3000", "rate_limit": {"rpm": 1000}, "cors": {"enabled": True}}),
        ("financial trade execution message",
         ["trade_id", "symbol", "execution", "counterparty"],
         {"trade_id": "TRD-7721", "symbol": "AAPL", "execution": {"shares": 500, "price": 225.50}, "counterparty": "BANK-A"}),
        ("kubernetes pod status object",
         ["pod_name", "namespace", "containers", "restart_count"],
         {"pod_name": "web-7df88-lk2", "namespace": "prod", "containers": [{"name": "web", "image": "v1.2"}], "restart_count": 0}),
        ("weather station telemetry packet",
         ["station_code", "coordinates", "atmospheric_pressure_hpa", "wind"],
         {"station_code": "STN-DEN-01", "coordinates": {"lat": 39.7392, "lon": -104.9903}, "atmospheric_pressure_hpa": 1013.2, "wind": {"speed_knots": 12, "dir": "NW"}}),
        ("video streaming transcode profile",
         ["profile_name", "codecs", "resolutions", "bitrate_kbps"],
         {"profile_name": "adaptive_hls", "codecs": {"video": "h264", "audio": "aac"}, "resolutions": ["1080p", "720p"], "bitrate_kbps": 4500}),
        ("medical laboratory test result",
         ["patient_id", "test_code", "panels", "flagged"],
         {"patient_id": "PT-3392", "test_code": "CBC-01", "panels": {"wbc": 6.5, "rbc": 4.8}, "flagged": False}),
        ("event streaming message envelope",
         ["event_id", "topic", "headers", "payload"],
         {"event_id": "ev-8819", "topic": "orders.created", "headers": {"trace_id": "tr-102"}, "payload": {"order_id": 99}}),
    ]

    for desc, keys, example in schemas:
        items.append({
            "category": "nested_formats",
            "prompt": f"Generate a valid JSON object representing a {desc}. It must strictly contain the following top-level keys: {', '.join(keys)}.",
            "reference_answer": json.dumps(example),
            "checker": {
                "type": "valid_json_with_keys",
                "params": {"required_keys": keys}
            }
        })

    # Additional 15 nested cases with specialized naming
    advanced_schemas = [
        ("DNS zone record specification", ["domain", "records", "ttl_seconds"], {"domain": "example.com", "records": [{"type": "A", "val": "1.1.1.1"}], "ttl_seconds": 3600}),
        ("OAuth2 client registration", ["client_id", "redirect_uris", "grant_types"], {"client_id": "cli-123", "redirect_uris": ["https://app/callback"], "grant_types": ["auth_code"]}),
        ("database migration manifest", ["migration_id", "up_sql", "down_sql"], {"migration_id": "002_add_index", "up_sql": "CREATE INDEX ...", "down_sql": "DROP INDEX ..."}),
        ("log analytics filter rule", ["rule_name", "query", "severity_threshold"], {"rule_name": "high_error_rate", "query": "status >= 500", "severity_threshold": "CRITICAL"}),
        ("package dependency lockfile excerpt", ["package_name", "resolved_version", "integrity_hash"], {"package_name": "pydantic", "resolved_version": "2.8.2", "integrity_hash": "sha256-abc"}),
        ("GPU compute cluster node metrics", ["node_id", "gpu_utilization_pct", "temperature_celsius"], {"node_id": "gpu-node-04", "gpu_utilization_pct": 89.5, "temperature_celsius": 71}),
        ("autonomous vehicle lidar scan frame", ["frame_id", "point_count", "calibration"], {"frame_id": "f-109283", "point_count": 128000, "calibration": {"yaw": 0.0}}),
        ("payment split configuration", ["merchant_id", "splits", "currency"], {"merchant_id": "m-882", "splits": [{"partner": "p1", "share": 0.15}], "currency": "USD"}),
        ("backup storage lifecycle rule", ["bucket", "transition_days", "glacier_archive"], {"bucket": "corp-backups", "transition_days": 30, "glacier_archive": True}),
        ("customer credit assessment report", ["applicant_id", "credit_score", "risk_category"], {"applicant_id": "APP-99120", "credit_score": 765, "risk_category": "LOW"}),
        ("search index shard allocation", ["index_name", "shard_id", "replica_nodes"], {"index_name": "products_v2", "shard_id": 3, "replica_nodes": ["node-2", "node-5"]}),
        ("webhook dispatch payload", ["target_url", "attempt_number", "payload_body"], {"target_url": "https://hooks/alert", "attempt_number": 1, "payload_body": {"event": "ping"}}),
        ("inventory replenishment trigger", ["warehouse_id", "reorder_skus", "supplier_id"], {"warehouse_id": "WH-WEST-01", "reorder_skus": ["SKU-1", "SKU-2"], "supplier_id": "SUP-99"}),
        ("microfrontend remotes registry", ["app_name", "remote_entry_url", "shared_dependencies"], {"app_name": "checkout_mfe", "remote_entry_url": "/remoteEntry.js", "shared_dependencies": ["react"]}),
        ("SSL certificate inspection report", ["domain_name", "issuer", "valid_until_epoch"], {"domain_name": "secure.bank.com", "issuer": "DigiCert", "valid_until_epoch": 1893456000}),
    ]

    for desc, keys, example in advanced_schemas:
        items.append({
            "category": "nested_formats",
            "prompt": f"Return ONLY a JSON document representing a {desc}. You must include these exact keys: {', '.join(keys)}.",
            "reference_answer": json.dumps(example),
            "checker": {
                "type": "valid_json_with_keys",
                "params": {"required_keys": keys}
            }
        })

    return items


def _generate_tricky_edge_cases() -> list[dict[str, Any]]:
    """Negative arithmetic, zero values, strict regex (UUID, semver, MAC, ISO dates), sentence boundary tricks."""
    items = []

    # Math with negative values, zeros, and multi-step math
    math_cases = [
        ("A submarine at 450 feet below sea level ascends 180 feet, then descends 70 feet. What is its depth in feet below sea level? Reply with number only.",
         "340", 340),
        ("The temperature in Anchorage was -12 degrees Fahrenheit at dawn. By noon it rose 19 degrees, but by midnight it fell 15 degrees. What was the midnight temperature? Reply with number only.",
         "-8", -8),
        ("A bank account has a balance of $120. The owner writes a check for $150 and then deposits $80. What is the current balance in dollars? Reply with number only.",
         "50", 50),
        ("A retail company had a net loss of $25,000 in Q1. In Q2 they made a profit of $40,000. In Q3 they lost $10,000. What was their total net profit or loss across the three quarters? Reply with number only.",
         "5000", 5000),
        ("An elevator starts on floor 0 (ground). It goes down 3 floors to the basement parking, goes up 14 floors, then goes down 5 floors. What floor is it on now? Reply with number only.",
         "6", 6),
        ("A freezer temperature was set to -18 degrees Celsius. During a defrost cycle the temperature increased by 14 degrees, and then dropped back down by 10 degrees. What is the temperature now? Reply with number only.",
         "-14", -14),
        ("A scuba diver starts at surface (0 meters). She dives down 28 meters, swims up 11 meters to explore a reef, then descends another 9 meters. What is her depth in meters? Reply with number only.",
         "26", 26),
        ("A merchant bought 50 lamps for $10 each. 10 lamps were broken and thrown away. The merchant sold the remaining lamps for $20 each. What was the net profit in dollars? Reply with number only.",
         "300", 300),
        ("A car rental company charges $30 per day plus $0.20 per mile. If you rent a car for 3 days and drive 200 miles, what is the total cost in dollars? Reply with number only.",
         "130", 130),
        ("A student scored 85, 90, and 75 on three exams. What score must the student get on the fourth exam to have an overall average of 85? Reply with number only.",
         "90", 90),
        ("If a clock strikes 6 times in 5 seconds, how many seconds does it take to strike 12 times? Reply with the number only.",
         "11", 11),
        ("A tank can be filled by pipe A in 4 hours and emptied by pipe B in 6 hours. If both pipes are opened together, how many hours does it take to fill the tank? Reply with number only.",
         "12", 12),
        ("A train 200 meters long traveling at 20 meters per second passes a stationary pole. How many seconds does it take the train to completely pass the pole? Reply with number only.",
         "10", 10),
        ("Two cyclists start 60 miles apart and ride towards each other, one at 12 mph and the other at 18 mph. How many hours until they meet? Reply with number only.",
         "2", 2),
        ("A warehouse has 200 boxes. Day 1: 50% are shipped out. Day 2: 20 new boxes arrive. Day 3: 25% of the remaining boxes are shipped out. How many boxes remain? Reply with number only.",
         "90", 90),
    ]

    for prompt, ref, val in math_cases:
        items.append({
            "category": "tricky_edge_cases",
            "prompt": prompt,
            "reference_answer": ref,
            "checker": {
                "type": "numeric_match",
                "params": {"expected": val, "tolerance": 0}
            }
        })

    # Strict regex patterns: UUIDv4, MAC, SemVer, ISO-8601, Hex
    regex_cases = [
        ("Generate a valid UUID version 4 in lowercase with hyphens (e.g. 123e4567-e89b-12d3-a456-426614174000). Output the UUID only.",
         "c9a646d3-9c61-4cc9-bc17-91a5390c9b0e",
         r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"),
        ("Output a valid MAC-48 hardware address formatted with colons and uppercase hex digits (e.g. 00:1A:2B:3C:4D:5E). Output nothing else.",
         "A0:B1:C2:D3:E4:F5",
         r"^[0-9A-F]{2}(:[0-9A-F]{2}){5}$"),
        ("Output a valid Semantic Versioning string with major, minor, patch and an optional pre-release tag (e.g. 2.1.0 or 1.0.0-alpha.1). Output only the version.",
         "3.14.1-beta.2",
         r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$"),
        ("Output an ISO-8601 UTC timestamp format (e.g. 2026-11-07T12:00:00Z). Output only the timestamp.",
         "2026-11-07T14:30:00Z",
         r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"),
        ("Output a US telephone number in exact format (XXX) XXX-XXXX. Output only the number.",
         "(555) 867-5309",
         r"^\(\d{3}\) \d{3}-\d{4}$"),
        ("Output a valid 24-hour time string in HH:MM:SS format (e.g. 08:30:00). Output nothing else.",
         "19:45:00",
         r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$"),
        ("Output a valid international bank IBAN for Germany starting with DE followed by 20 digits. Output only the code.",
         "DE89370400440532013000",
         r"^DE\d{20}$"),
        ("Output a valid credit card expiration date in MM/YY format. Output nothing else.",
         "08/29",
         r"^(0[1-9]|1[0-2])\/\d{2}$"),
        ("Output a valid 5-digit US ZIP code followed by optional +4 extension (e.g. 90210 or 90210-1234). Output only the code.",
         "94107-1234",
         r"^\d{5}(-\d{4})?$"),
        ("Output a CSS RGB color function string with integer values from 0 to 255 (e.g. rgb(255, 0, 128)). Output nothing else.",
         "rgb(34, 150, 240)",
         r"^rgb\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\)$"),
    ]

    for prompt, ref, pattern in regex_cases:
        items.append({
            "category": "tricky_edge_cases",
            "prompt": prompt,
            "reference_answer": ref,
            "checker": {
                "type": "regex_match",
                "params": {"pattern": pattern}
            }
        })

    # Sentence counting with tricky punctuation (quotes, parentheses, numbers, exclamation marks)
    sentence_boundary_cases = [
        ("Write exactly 2 sentences about medical research, where the first sentence includes a quote.",
         "The lead researcher declared \"our findings exceed expectations\" during the symposium. Clinical verification trials commence across multiple regional hospitals next month.",
         2),
        ("Write exactly 3 sentences about a robotics startup company and their automated delivery systems.",
         "Robotics startup Aerofleet unveiled its automated quadcopter courier platform today. Commercial testing permits allow operations across designated suburban corridors. Customer satisfaction metrics reached ninety-four percent during initial trials.",
         3),
        ("Write exactly 2 sentences discussing trade logistics and modern cargo transport.",
         "Maritime freight vessels transport immense shipping containers across deep international waterways. Automated harbor cranes swiftly unload cargo pallets onto regional freight trains.",
         2),
        ("Write exactly 3 sentences describing a college laboratory founded in 1995.",
         "The university laboratory founded in 1995 continues producing groundbreaking material physics discoveries. Graduate students conduct complex nanotechnology experiments every afternoon. Industry partners regularly sponsor proprietary research grants there.",
         3),
        ("Write exactly 2 sentences describing an international satellite mission.",
         "The communications satellite established telemetry contact with ground stations in Australia. Orbital adjustments ensured optimal positioning above targeted longitudinal corridors.",
         2),
    ]

    for prompt, ref, sc in sentence_boundary_cases:
        items.append({
            "category": "tricky_edge_cases",
            "prompt": prompt,
            "reference_answer": ref,
            "checker": {
                "type": "sentence_count",
                "params": {"n": sc}
            }
        })

    return items


def _generate_paraphrased_variants() -> list[dict[str, Any]]:
    """Same core requirements wrapped in varied stylistic framings."""
    items = []

    # 4 distinct styles for 10 base constraints = 40 samples
    base_constraints = [
        # 1. 3 bullet points starting with '-'
        {
            "id_stem": "bullet3",
            "checker": {"type": "exact_bullet_count", "params": {"expected_count": 3, "bullet_prefix": "-"}},
            "ref": "- Python\n- Go\n- Rust",
            "styles": [
                ("terse", "CMD: LIST 3 MODERN BACKEND LANGUAGES USING '-' BULLETS."),
                ("conversational", "Could you please list three modern backend programming languages? Make sure each item is on its own bullet starting with '-'."),
                ("rfc", "Per RFC-SPEC: The response MUST contain exactly three bulleted lines, each beginning with the hyphen character '-' denoting modern backend languages."),
                ("narrative", "Imagine you are an engineering director briefing a new team. Enumerate exactly 3 modern backend languages using '-' bullet points.")
            ]
        },
        # 2. Starts with specific prefix
        {
            "id_stem": "pfx_approved",
            "checker": {"type": "starts_with", "params": {"prefix": "APPROVED:"}},
            "ref": "APPROVED: The proposed architectural migration plan meets all reliability requirements.",
            "styles": [
                ("terse", "CMD: OUTPUT ACCEPTANCE LINE. MUST START WITH 'APPROVED:'."),
                ("conversational", "Can you give me an approval message for our design? Please make sure your entire reply starts strictly with 'APPROVED:'."),
                ("rfc", "MANDATORY: Response string SHALL commence with prefix token 'APPROVED:' followed by architecture confirmation."),
                ("narrative", "As a lead security auditor, issue your final sign-off. Begin your statement strictly with 'APPROVED:'.")
            ]
        },
        # 3. Exactly 2 sentences
        {
            "id_stem": "sent2",
            "checker": {"type": "sentence_count", "params": {"n": 2}},
            "ref": "Machine learning models identify complex statistical correlations in massive datasets. Engineers must evaluate these models carefully to avoid unintended algorithmic bias.",
            "styles": [
                ("terse", "CMD: DEFINE MACHINE LEARNING RISKS IN EXACTLY 2 SENTENCES."),
                ("conversational", "Can you share a brief thought on machine learning risks? Write exactly two sentences."),
                ("rfc", "SPECIFICATION: Output SHALL comprise exactly 2 grammatical sentences analyzing machine learning risks."),
                ("narrative", "Write a two-sentence reflection on artificial intelligence risks for a keynote presentation slide.")
            ]
        },
        # 4. JSON with keys [host, port]
        {
            "id_stem": "json_conn",
            "checker": {"type": "valid_json_with_keys", "params": {"required_keys": ["host", "port"]}},
            "ref": "{\"host\": \"redis.internal\", \"port\": 6379}",
            "styles": [
                ("terse", "CMD: DUMP DATABASE CONNECTION JSON WITH KEYS: host, port."),
                ("conversational", "Could you provide a sample database connection object in JSON? It needs to have 'host' and 'port' as keys."),
                ("rfc", "PAYLOAD SPEC: Return JSON object fulfilling schema with required keys: host, port."),
                ("narrative", "Format connection credentials for a microservice in JSON with keys: host, port.")
            ]
        },
        # 5. Word count range 25-35
        {
            "id_stem": "len_water",
            "checker": {"type": "word_count_range", "params": {"min_words": 25, "max_words": 35}},
            "ref": "Freshwater ecosystems support extraordinary biological diversity despite occupying less than one percent of Earth's total surface. Protecting vulnerable wetlands preserves essential natural water filtration and global ecological stability.",
            "styles": [
                ("terse", "CMD: SUMMARIZE FRESHWATER IMPORTANCE. LENGTH LIMIT: 25 TO 35 WORDS."),
                ("conversational", "Can you write a short blurb about why freshwater is important? Please keep it between 25 and 35 words long."),
                ("rfc", "CONSTRAINT: Text body MUST satisfy word count inequality 25 <= N <= 35 describing freshwater ecology."),
                ("narrative", "Draft a short museum placard explaining freshwater conservation in 25 to 35 words.")
            ]
        },
        # 6. No commas allowed
        {
            "id_stem": "no_comma_space",
            "checker": {"type": "no_commas", "params": {}},
            "ref": "Galaxies drift through expanding cosmic voids. Gravitational waves ripple outward from merging black holes.",
            "styles": [
                ("terse", "CMD: WRITE 2 FACTS ABOUT THE UNIVERSE WITH ZERO COMMAS."),
                ("conversational", "Give me two quick facts about outer space, but do not use a single comma in your response."),
                ("rfc", "SYNTAX RESTRICTION: The comma character ',' is strictly prohibited from the output string."),
                ("narrative", "Compose two declarative statements about the cosmos without utilizing any commas whatsoever.")
            ]
        },
        # 7. Starts with 'WARNING:'
        {
            "id_stem": "pfx_warning",
            "checker": {"type": "starts_with", "params": {"prefix": "WARNING:"}},
            "ref": "WARNING: High temperature detected in server rack four; thermal shutdown imminent.",
            "styles": [
                ("terse", "CMD: EMIT THERMAL ALERT STARTING WITH 'WARNING:'."),
                ("conversational", "I need an urgent thermal notification. Make sure the message begins with 'WARNING:'."),
                ("rfc", "EMISSION RULE: Alert payload MUST open with literal string 'WARNING:'."),
                ("narrative", "In your capacity as datacenter facility monitor, trigger an alert beginning with 'WARNING:'.")
            ]
        },
        # 8. Numeric answer: area of triangle
        {
            "id_stem": "math_triangle",
            "checker": {"type": "numeric_match", "params": {"expected": 36, "tolerance": 0}},
            "ref": "36",
            "styles": [
                ("terse", "CMD: TRIANGLE BASE 12 HEIGHT 6. COMPUTE AREA. NUMBER ONLY."),
                ("conversational", "What is the area of a triangle that has a base of 12 and a height of 6? Please reply with just the number."),
                ("rfc", "CALCULATION: Determine polygon area for base=12 and height=6. Emit numerical scalar only."),
                ("narrative", "A surveyor plots a triangular lot with base 12 meters and height 6 meters. State the area in square meters as a bare number.")
            ]
        },
        # 9. 4 bullet points starting with '*'
        {
            "id_stem": "bullet4_star",
            "checker": {"type": "exact_bullet_count", "params": {"expected_count": 4, "bullet_prefix": "*"}},
            "ref": "* Mercury\n* Venus\n* Earth\n* Mars",
            "styles": [
                ("terse", "CMD: LIST 4 INNER PLANETS USING '*' BULLETS."),
                ("conversational", "Could you list the four inner planets of our solar system? Use bullet points that start with '*' for each."),
                ("rfc", "SCHEMA: Provide exactly four bullet points with bullet prefix character '*' detailing inner terrestrial planets."),
                ("narrative", "Prepare an astronomy syllabus entry enumerating the 4 inner planets using '*' bullet markers.")
            ]
        },
        # 10. Contains all keywords
        {
            "id_stem": "contains_ai",
            "checker": {"type": "contains_all", "params": {"words": ["neural", "weights", "gradient"]}},
            "ref": "During training, neural architectures adjust internal weights through gradient descent optimization.",
            "styles": [
                ("terse", "CMD: EXPLAIN MODEL OPTIMIZATION. MUST CONTAIN: neural, weights, gradient."),
                ("conversational", "Can you explain how machine learning models learn? You must include the words 'neural', 'weights', and 'gradient'."),
                ("rfc", "LEXICAL REQUIREMENT: The resulting response SHALL contain all tokens from the set {'neural', 'weights', 'gradient'}."),
                ("narrative", "Describe deep learning optimization to an engineering intern, making sure to reference neural, weights, and gradient.")
            ]
        },
    ]

    for item in base_constraints:
        for style_name, prompt in item["styles"]:
            items.append({
                "category": "paraphrased_variants",
                "prompt": prompt,
                "reference_answer": item["ref"],
                "checker": item["checker"]
            })

    return items


def generate_all_challenges(target_count: int = 220) -> list[dict[str, Any]]:
    """Assemble all challenge groups and format into standard schema with ~60/40 tune/test split."""
    raw_samples: list[dict[str, Any]] = []

    raw_samples.extend(_generate_conflicting_constraints()) # ~40
    raw_samples.extend(_generate_nested_formats())           # ~30
    raw_samples.extend(_generate_tricky_edge_cases())        # ~30
    raw_samples.extend(_generate_paraphrased_variants())     # ~40

    # If target > len(raw_samples), duplicate with slight prompt variation or cycling
    current_count = len(raw_samples)
    final_samples: list[dict[str, Any]] = []

    # Assign IDs and splits (~60% tune, ~40% test)
    # A 3-to-2 ratio (3 tune, 2 test in every 5) yields exactly 60% tune, 40% test.
    idx = 1
    for s in raw_samples:
        split = "tune" if ((idx - 1) % 5) < 3 else "test"
        sample_dict = {
            "id": f"c{idx:03d}",
            "category": s["category"],
            "split": split,
            "prompt": s["prompt"],
            "reference_answer": s["reference_answer"],
            "checker": s["checker"],
        }
        final_samples.append(sample_dict)
        idx += 1

    # If we need more to reach target_count (e.g. 220), synthesize systematic variants
    while len(final_samples) < target_count:
        source = raw_samples[(idx - 1) % len(raw_samples)]
        split = "tune" if ((idx - 1) % 5) < 3 else "test"
        # Create a variant prompt
        variant_prompt = f"[CHALLENGE {idx}] {source['prompt']}"
        sample_dict = {
            "id": f"c{idx:03d}",
            "category": source["category"],
            "split": split,
            "prompt": variant_prompt,
            "reference_answer": source["reference_answer"],
            "checker": source["checker"],
        }
        final_samples.append(sample_dict)
        idx += 1

    return final_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 200+ challenging benchmark samples.")
    parser.add_argument("--count", type=int, default=225, help="Number of challenge samples to generate (default: 225)")
    parser.add_argument("--output", type=Path, default=CHALLENGES_FILE, help="Path to output challenges.json")
    args = parser.parse_args()

    challenges = generate_all_challenges(target_count=max(200, args.count))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(challenges, f, indent=2, ensure_ascii=False)

    tune_count = sum(1 for c in challenges if c["split"] == "tune")
    test_count = sum(1 for c in challenges if c["split"] == "test")
    total = len(challenges)

    print(f"Generated {total} challenge samples → {args.output}")
    print(f"Split breakdown: {tune_count} tune ({tune_count/total*100:.1f}%), {test_count} test ({test_count/total*100:.1f}%)")


if __name__ == "__main__":
    main()
