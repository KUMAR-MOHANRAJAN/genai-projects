"""Ground truth: test queries with expected answers and retrieval keywords.

Used by:
  - Evaluator: compare pipeline answer to expected_answer (correctness scoring)
  - Retrieval metrics: check if retrieved chunks contain keywords (precision/recall proxy)
  - Optimizer loop: the target the system optimizes toward

Keywords are strategy-agnostic — they work across fixed_size, recursive_split,
and semantic chunking without needing per-version chunk ID mapping.

Golden set: 25 questions grounded in the 9-document corpus (corpus/{hr,technical,
legal,finance}/), 4 domains. Mix of 11 easy / 10 medium single-domain questions
plus 4 hard cross-domain questions that require retrieving chunks from two
different documents to answer completely — a genuine test of retrieval breadth,
not just chunk quality.
"""

TEST_QUERIES = [
    # ── HR (6) ────────────────────────────────────────────────────────────
    (
        "How many days of annual leave does an employee with 5 years of tenure get?",
        "Employees with 3 years to less than 7 years of tenure are entitled to "
        "21 working days of paid annual leave per year.",
        ["annual leave", "21", "days", "tenure", "entitlement"],
    ),
    (
        "How many paid sick leave days do employees get per year?",
        "All full-time employees receive 12 days of paid sick leave per year. "
        "Sick leave does not accumulate or carry over between calendar years.",
        ["sick leave", "12", "days", "year", "paid"],
    ),
    (
        "How long is maternity leave and at what percentage of salary is it paid?",
        "The primary caregiver is entitled to 26 weeks of paid maternity leave, "
        "paid at 100% of basic salary.",
        ["maternity", "26 weeks", "100%", "salary", "leave"],
    ),
    (
        "What is the monthly wellness allowance and how is it paid?",
        "Employees receive a monthly wellness allowance of SGD 80 credited to "
        "their CXA employee benefits wallet.",
        ["wellness", "allowance", "SGD 80", "monthly", "CXA"],
    ),
    (
        "What are the gift and hospitality limits under the Code of Conduct?",
        "Gifts up to SGD 50 in value and hospitality up to SGD 150 per occasion "
        "may be accepted or offered; anything above these limits requires prior "
        "approval from Legal.",
        ["gift", "SGD 50", "hospitality", "SGD 150", "approval"],
    ),
    (
        "What is the childcare subsidy amount and eligibility?",
        "Employees with children below age 7 receive a monthly childcare "
        "subsidy of SGD 150 per child, up to a maximum of 2 children.",
        ["childcare", "subsidy", "SGD 150", "age 7", "children"],
    ),

    # ── Technical (5) ─────────────────────────────────────────────────────
    (
        "What is the SEV-1 response time requirement?",
        "SEV-1 (Critical) incidents require a 15-minute response time, with "
        "immediate escalation to the Engineering Lead and CTO.",
        ["SEV-1", "15 minutes", "response", "critical", "escalate"],
    ),
    (
        "What is the LLM provider failover order when the primary fails?",
        "The system automatically fails over from Gemini Flash 3.5 to KiMi K2.6 "
        "to Gemini 3.5.",
        ["Gemini Flash", "KiMi", "failover", "LLM provider", "Gemini 3.5"],
    ),
    (
        "What are the steps to manually promote a Redis replica during an outage?",
        "Check ElastiCache status in the AWS Console; if automatic replica "
        "promotion fails after 60 seconds, manually promote a replica using "
        "aws elasticache modify-replication-group, then update the REDIS_URL "
        "secret and force ECS tasks to restart.",
        ["ElastiCache", "replica", "promote", "Redis", "modify-replication-group"],
    ),
    (
        "What Python and Node.js versions are required for local development?",
        "Python 3.11 and Node.js 20 or later are required, managed via pyenv "
        "and nvm respectively.",
        ["Python 3.11", "Node.js 20", "pyenv", "nvm", "prerequisites"],
    ),
    (
        "What must a post-mortem include and within what time frame?",
        "All SEV-1 and SEV-2 incidents require a written post-mortem within 48 "
        "hours of resolution, including timeline, root cause, impact, "
        "remediation steps, and action items to prevent recurrence.",
        ["post-mortem", "48 hours", "root cause", "timeline", "action items"],
    ),

    # ── Legal (5) ─────────────────────────────────────────────────────────
    (
        "What is the late payment interest rate in the vendor contract?",
        "Late payment incurs interest at 1.5% per month on the outstanding "
        "balance, applied from the due date until payment.",
        ["late payment", "1.5%", "interest", "vendor", "invoice"],
    ),
    (
        "How long do confidentiality obligations survive after the vendor contract terminates?",
        "Confidentiality obligations survive termination for 5 years, or "
        "indefinitely for information that remains a trade secret.",
        ["confidentiality", "5 years", "survive", "termination", "trade secret"],
    ),
    (
        "What is the NDA term and how long do obligations survive after termination?",
        "The NDA runs for a 2-year term, and confidentiality obligations "
        "survive termination for an additional 3 years (or indefinitely for "
        "trade secrets).",
        ["NDA", "2-year", "3-year", "survive", "term"],
    ),
    (
        "Within what time frame must a vendor notify of a personal data breach?",
        "The vendor must notify the company within 24 hours of becoming aware "
        "of any personal data breach.",
        ["data breach", "24 hours", "notify", "vendor", "personal data"],
    ),
    (
        "What law governs the NDA and where is jurisdiction?",
        "The NDA is governed by the laws of Singapore, with exclusive "
        "jurisdiction in the courts of Singapore.",
        ["Singapore", "governing law", "jurisdiction", "courts", "NDA"],
    ),

    # ── Finance (5) ───────────────────────────────────────────────────────
    (
        "What is the domestic hotel expense limit per night?",
        "Domestic accommodation is reimbursable up to SGD 250 per night in "
        "Singapore.",
        ["hotel", "SGD 250", "night", "domestic", "accommodation"],
    ),
    (
        "What is the international accommodation limit in Tier 1 cities?",
        "International accommodation is reimbursable up to SGD 350 per night "
        "in Tier 1 cities such as New York, London, Tokyo, and Sydney.",
        ["international", "SGD 350", "Tier 1", "accommodation", "night"],
    ),
    (
        "What approval is required for an expense of SGD 5,000?",
        "Expenses between SGD 2,001 and SGD 10,000 require Vice President "
        "(VP) approval.",
        ["approval", "VP", "SGD 5,000", "expense", "matrix"],
    ),
    (
        "What is the FY2026 total revenue target?",
        "The FY2026 revenue target is SGD 18 million.",
        ["FY2026", "revenue", "SGD 18 million", "target", "budget"],
    ),
    (
        "What is the FY2026 cloud infrastructure budget and how is it split?",
        "The FY2026 cloud infrastructure budget is SGD 720,000, split across "
        "AWS ECS Fargate (SGD 310,000), Supabase (SGD 180,000), the EURI API "
        "gateway and LLM provider costs (SGD 180,000), and monitoring/CDN "
        "(SGD 50,000).",
        ["cloud infrastructure", "SGD 720,000", "AWS", "Supabase", "budget"],
    ),

    # ── Cross-domain / hard (4) — requires chunks from two different docs ──
    (
        "What is the maternity leave duration, and what is the SEV-1 incident response time?",
        "Maternity leave is 26 weeks, paid at 100% of basic salary. "
        "Separately, SEV-1 (Critical) incidents require a 15-minute response time.",
        ["maternity", "26 weeks", "SEV-1", "15 minutes", "response"],
    ),
    (
        "What is the late payment penalty in the vendor contract, and what expense amount requires CFO approval?",
        "The vendor contract charges 1.5% per month interest on late "
        "payments. Separately, any expense above SGD 10,000 requires CFO "
        "approval.",
        ["late payment", "1.5%", "CFO", "SGD 10,000", "approval"],
    ),
    (
        "What is the gift limit in the Code of Conduct, and what is the data breach notification window in the vendor contract?",
        "The Code of Conduct sets a gift limit of SGD 50. Separately, the "
        "vendor contract requires personal data breach notification within "
        "24 hours.",
        ["gift", "SGD 50", "data breach", "24 hours", "notification"],
    ),
    (
        "What is the FY2026 revenue target, and what is the NDA's confidentiality survival period?",
        "The FY2026 revenue target is SGD 18 million. Separately, the NDA's "
        "confidentiality obligations survive termination for 3 years (2-year "
        "base term).",
        ["FY2026", "SGD 18 million", "NDA", "3 years", "confidentiality"],
    ),
]


def load_ground_truth() -> list[tuple[str, str, list[str]]]:
    """Return list of (question, expected_answer, keywords) tuples."""
    return TEST_QUERIES


def get_query(index: int = 0) -> tuple[str, str, list[str]]:
    """Get a specific test query by index."""
    return TEST_QUERIES[index]
