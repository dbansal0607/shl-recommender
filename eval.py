# eval.py
"""
Measures Recall@10 on the 10 public conversation traces.
Run: python eval.py (while server is running)
"""
import requests, time

BASE = "http://localhost:8000"

TRACES = [
    {
        "name": "C1 - Senior Leadership Selection",
        "turns": [
            "We need a solution for senior leadership.",
            "The pool consists of CXOs, director-level positions; people with more than 15 years of experience.",
            "Selection — comparing candidates against a leadership benchmark.",
            "Perfect, that's what we need.",
        ],
        "expected": ["Occupational Personality Questionnaire OPQ32r", "OPQ Leadership Report", "OPQ Universal Competency Report 2.0"],
    },
    {
        "name": "C2 - Senior Rust Engineer",
        "turns": [
            "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?",
            "Yes, go ahead. Should I also add a cognitive test for this level?",
            "That works. Thanks.",
        ],
        "expected": ["Smart Interview Live Coding", "Linux Programming (General)", "Networking and Implementation (New)", "SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r"],
    },
    {
        "name": "C3 - Contact Centre 500 Agents",
        "turns": [
            "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus.",
            "English.",
            "US.",
            "Perfect — new simulation for volume, old solution for finalists. Confirmed.",
        ],
        "expected": ["SVAR - Spoken English (US) (New)", "Contact Center Call Simulation (New)", "Customer Service Phone Simulation", "Entry Level Customer Serv-Retail & Contact Center"],
    },
    {
        "name": "C4 - Graduate Financial Analysts",
        "turns": [
            "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.",
            "Good. Can you also add a situational judgement element — work-context decision making for graduates?",
            "That covers it.",
        ],
        "expected": ["SHL Verify Interactive – Numerical Reasoning", "Financial Accounting (New)", "Graduate Scenarios", "Occupational Personality Questionnaire OPQ32r"],
    },
    {
        "name": "C5 - Sales Reskilling Audit",
        "turns": [
            "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?",
            "Clear. We'll use OPQ for everyone and add MQ only where we want motivators.",
        ],
        "expected": ["Global Skills Assessment", "Global Skills Development Report", "Occupational Personality Questionnaire OPQ32r", "OPQ MQ Sales Report", "Sales Transformation 2.0 - Individual Contributor"],
    },
    {
        "name": "C6 - Plant Operators Safety",
        "turns": [
            "We're hiring plant operators for a chemical facility. Safety is absolute top priority.",
            "We're industrial. The 8.0 bundle is the right fit. Confirmed.",
        ],
        "expected": ["Manufac. & Indust. - Safety & Dependability 8.0", "Dependability and Safety Instrument (DSI)", "Workplace Health and Safety (New)"],
    },
    {
        "name": "C7 - Bilingual Healthcare Admin",
        "turns": [
            "We're hiring bilingual healthcare admin staff in South Texas. HIPAA compliance is critical.",
            "They're functionally bilingual — English fluent for written work. Go with the hybrid.",
            "Understood. Keep the shortlist as-is.",
        ],
        "expected": ["HIPAA (Security)", "Medical Terminology (New)", "Dependability and Safety Instrument (DSI)", "Occupational Personality Questionnaire OPQ32r"],
    },
    {
        "name": "C8 - Admin Assistants Excel Word",
        "turns": [
            "I need to quickly screen admin assistants for Excel and Word daily.",
            "In that case, I am OK with adding a simulation.",
            "That's good.",
        ],
        "expected": ["MS Excel (New)", "MS Word (New)", "Microsoft Excel 365 (New)", "Microsoft Word 365 (New)", "Occupational Personality Questionnaire OPQ32r"],
    },
    {
        "name": "C9 - Senior Full-Stack Engineer",
        "turns": [
            'Here\'s the JD: "Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL, AWS, Docker. Will mentor engineers and contribute to architecture."',
            "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant.",
            "Senior IC.",
            "Add AWS and Docker. Drop REST.",
            "Keep Verify G+. Locking it in.",
        ],
        "expected": ["Core Java (Advanced Level) (New)", "Spring (New)", "SQL (New)", "Amazon Web Services (AWS) Development (New)", "Docker (New)", "SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r"],
    },
    {
        "name": "C10 - Graduate Management Trainee",
        "turns": [
            "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement.",
            "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios.",
        ],
        "expected": ["SHL Verify Interactive G+", "Graduate Scenarios"],
    },
]


def recall_at_k(got: list[str], expected: list[str], k: int = 10) -> float:
    if not expected:
        return 1.0
    top_k = got[:k]
    hits = sum(
        1 for exp in expected
        if any(exp.lower() in rec.lower() or rec.lower() in exp.lower() for rec in top_k)
    )
    return hits / len(expected)


def run_trace(trace: dict) -> float:
    history = []
    final_recs = []
    print(f"\n{'─'*60}")
    print(f"Running: {trace['name']}")

    for turn in trace["turns"]:
        history.append({"role": "user", "content": turn})
        try:
            time.sleep(3)
            resp = requests.post(
                f"{BASE}/chat",
                json={"messages": history},
                timeout=120,
            ).json()
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            break

        # Validate schema
        assert "reply"               in resp, "Missing 'reply'"
        assert "recommendations"     in resp, "Missing 'recommendations'"
        assert "end_of_conversation" in resp, "Missing 'end_of_conversation'"
        for rec in resp["recommendations"]:
            assert "name"      in rec, "Rec missing 'name'"
            assert "url"       in rec, "Rec missing 'url'"
            assert "test_type" in rec, "Rec missing 'test_type'"
            assert "shl.com/products/product-catalog" in rec["url"], f"Invalid URL: {rec['url']}"

        reply = resp["reply"]
        history.append({"role": "assistant", "content": reply})

        if resp["recommendations"]:
            final_recs = [r["name"] for r in resp["recommendations"]]

        print(f"  Turn {len(history)//2}: recs={len(resp['recommendations'])} | eoc={resp['end_of_conversation']}")

        if resp["end_of_conversation"]:
            break

    score = recall_at_k(final_recs, trace["expected"])
    status = "✅" if score >= 0.5 else "⚠️"
    print(f"  {status} Recall@10 = {score:.2f}")
    print(f"  Got:      {final_recs[:5]}")
    print(f"  Expected: {trace['expected'][:5]}")
    return score


if __name__ == "__main__":
    print("Starting evaluation against 10 public traces…\n")
    scores = [run_trace(t) for t in TRACES]
    mean   = sum(scores) / len(scores)

    print(f"\n{'='*60}")
    print(f"MEAN Recall@10: {mean:.3f}")
    print(f"{'='*60}")
    for t, s in zip(TRACES, scores):
        bar = "█" * int(s * 10)
        print(f"  {s:.2f} {bar:<10} {t['name']}")

    print(f"\n→ Put this number in your approach doc: {mean:.3f}")