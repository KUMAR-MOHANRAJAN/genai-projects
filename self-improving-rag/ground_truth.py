"""Ground truth: test queries with expected answers and retrieval keywords.

Used by:
  - Evaluator: compare pipeline answer to expected_answer (correctness scoring)
  - Retrieval metrics: check if retrieved chunks contain keywords (precision/recall proxy)
  - Optimizer loop: the target the system optimizes toward

Keywords are strategy-agnostic — they work across fixed_size, recursive_split,
and semantic chunking without needing per-version chunk ID mapping.

Golden set: 23 questions grounded in the 10-document Canadian workplace corpus
(corpus/{hr,technical,legal,finance}/), 4 domains. Mix of easy, medium,
and hard cross-domain questions that require retrieving chunks from two
different documents to answer completely — a genuine test of retrieval breadth,
not just chunk quality.
"""

_LEGACY_TEST_QUERIES = [
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


# Canada-focused golden set for the active organizational policy corpus.
TEST_QUERIES = [
    # ── HR / federal employment baseline (6) ─────────────────────────────
    (
        "How many paid medical-leave days can a federally regulated employee earn each year?",
        "Employees can earn up to 10 paid medical-leave days per calendar year: 3 days after 30 days of continuous employment, then 1 additional day per completed month.",
        ["medical leave", "10", "30 days", "completed month", "paid"],
    ),
    (
        "What vacation entitlement does a NorthStar employee receive after five years of service?",
        "Employees with 5 or more years of continuous service receive 20 working days of paid vacation per year.",
        ["5 years", "20 working days", "vacation", "service"],
    ),
    (
        "What maternity leave and company top-up are available to an eligible employee?",
        "Eligible employees may take up to 17 weeks of unpaid maternity leave, and NorthStar provides up to 8 weeks at 75% of regular base pay for eligible EI recipients.",
        ["17 weeks", "8 weeks", "75%", "maternity", "Employment Insurance"],
    ),
    (
        "How much is the monthly wellness allowance?",
        "Employees receive a CAD 75 monthly wellness allowance for fitness, wellbeing, or ergonomic home-office expenses.",
        ["CAD 75", "monthly", "wellness", "ergonomic"],
    ),
    (
        "What is NorthStar's RRSP matching benefit?",
        "NorthStar matches employee RRSP contributions dollar for dollar up to 4% of base salary after 3 months of service.",
        ["RRSP", "4%", "3 months", "matches"],
    ),
    (
        "Which federal general holidays are included in the statutory holiday policy?",
        "The policy lists New Year's Day, Good Friday, Victoria Day, Canada Day, Labour Day, National Day for Truth and Reconciliation, Thanksgiving, Remembrance Day, Christmas Day, and Boxing Day.",
        ["Canada Day", "Labour Day", "Remembrance Day", "Boxing Day", "general holidays"],
    ),
    # ── Finance / legal / technical (8) ──────────────────────────────────
    (
        "What approval is required for a CAD 4,500 expense?",
        "An expense between CAD 3,001 and CAD 12,000 requires VP approval.",
        ["CAD 4,500", "CAD 3,001", "CAD 12,000", "VP", "approval"],
    ),
    (
        "What is the Canadian domestic hotel limit per night?",
        "Accommodation for domestic Canadian travel is reimbursable up to CAD 275 per night.",
        ["Canada", "accommodation", "CAD 275", "night"],
    ),
    (
        "What is the annual learning and development budget?",
        "Each employee has a CAD 2,000 annual learning budget for role-relevant development.",
        ["CAD 2,000", "annual", "learning", "development"],
    ),
    (
        "What is the late-payment interest rate in the vendor agreement?",
        "Late payment incurs interest at 1.5% per month on the outstanding balance.",
        ["late payment", "1.5%", "per month", "vendor"],
    ),
    (
        "When must a vendor report a personal data breach?",
        "A vendor must notify the Company within 24 hours of becoming aware of a personal data breach.",
        ["vendor", "24 hours", "data breach", "notify"],
    ),
    (
        "What law governs the NDA?",
        "The NDA is governed by Ontario law and the applicable federal laws of Canada, with Ontario courts having exclusive jurisdiction.",
        ["Ontario", "federal laws", "Canada", "jurisdiction", "NDA"],
    ),
    (
        "What is the SEV-1 response-time requirement?",
        "A SEV-1 critical incident requires a 15-minute response and immediate escalation to the Engineering Lead and CTO.",
        ["SEV-1", "15 minutes", "Engineering Lead", "CTO"],
    ),
    (
        "When is a post-mortem required after an incident?",
        "All SEV-1 and SEV-2 incidents require a written post-mortem within 48 hours of resolution, including timeline, root cause, Canadian customer impact, remediation, and prevention actions.",
        ["SEV-1", "SEV-2", "48 hours", "post-mortem", "customer impact"],
    ),
    # ── Medium and hard cross-document scenarios (4) ─────────────────────
    # TEST: Medium, 2 sources - statutory_holidays_policy + leave_policy.
    (
        "A federal general holiday falls during approved vacation. What happens to the employee's vacation balance?",
        "The employee does not use a vacation day for the holiday; it is recorded separately and the vacation balance remains available.",
        ["general holiday", "vacation", "recorded separately", "balance"],
    ),
    # TEST: Hard, 2 sources - code_of_conduct + leave_policy.
    (
        "An employee has a suspected customer-data breach and needs medical leave. Who must receive the breach report, how quickly, and what paid leave may be available?",
        "The employee must report the suspected breach to the Privacy Officer and Security team within 2 hours. They may earn up to 10 paid medical-leave days per year under the federal baseline.",
        ["Privacy Officer", "Security", "2 hours", "medical leave", "10"],
    ),
    # TEST: Medium, 2 sources - benefits_guide + expense_policy.
    (
        "An employee wants a CAD 600 home-office monitor and is also claiming a monthly wellness expense. Which benefits apply?",
        "The CAD 600 one-time home-office equipment allowance applies to hybrid employees, while the separate CAD 75 monthly wellness allowance can cover eligible ergonomic expenses.",
        ["CAD 600", "home-office", "CAD 75", "wellness", "hybrid"],
    ),
    # TEST: Medium, 2 sources - budget_guidelines_2026 + expense_policy.
    (
        "What is the company's FY2026 revenue target, and what approval is needed for a CAD 13,000 business expense?",
        "The FY2026 revenue target is CAD 24 million. A business expense above CAD 12,000 requires CFO approval.",
        ["FY2026", "CAD 24 million", "CAD 13,000", "CAD 12,000", "CFO"],
    ),
    # TEST: Hard, 2 sources - leave_policy + production_runbook.
    (
        "What maternity leave and company top-up are available to an eligible employee, and what is the SEV-1 incident response time?",
        "An eligible employee may take up to 17 weeks of unpaid maternity leave, with a NorthStar supplementary benefit of up to 8 weeks at 75% of regular base pay for eligible EI recipients. Separately, a SEV-1 critical incident requires a 15-minute response and immediate escalation to the Engineering Lead and CTO.",
        ["maternity", "17 weeks", "8 weeks", "75%", "SEV-1", "15 minutes", "Engineering Lead", "CTO"],
    ),
    # ── Hard three-source scenarios (3) ──────────────────────────────────
    # TEST: Hard, 3 sources - vendor_contract_template + code_of_conduct + production_runbook.
    (
        "A vendor reports a possible customer-data breach during a SEV-1 incident. What notification and incident-response deadlines apply, and what must the post-mortem cover?",
        "The vendor must notify NorthStar within 24 hours of becoming aware of the breach. Internally, employees must report suspected breaches to the Privacy Officer and Security team within 2 hours. A SEV-1 requires a 15-minute response, and the post-mortem is due within 48 hours with the timeline, root cause, Canadian customer impact, remediation, and prevention actions.",
        ["vendor", "24 hours", "Privacy Officer", "Security", "2 hours", "SEV-1", "15 minutes", "48 hours", "post-mortem"],
    ),
    # TEST: Hard, 2 sources - benefits_guide + expense_policy.
    (
        "A hybrid employee is attending a role-relevant conference in Canada and needs a CAD 600 monitor for their home office. What learning, travel, and equipment benefits or limits apply?",
        "The employee can use the CAD 2,000 annual learning budget for the role-relevant conference. Domestic accommodation is reimbursable up to CAD 275 per night and domestic meals up to CAD 90 per day. A hybrid employee can use the one-time CAD 600 home-office equipment allowance, with IT approval for security-sensitive equipment.",
        ["CAD 2,000", "learning budget", "CAD 275", "accommodation", "CAD 90", "meals", "CAD 600", "hybrid", "IT approval"],
    ),
    # TEST: Hard, 2 sources - code_of_conduct + vendor_contract_template.
    (
        "An employee learns that a contractor may have disclosed confidential customer information. What internal reporting path and timing apply, what must the vendor do, and how long do confidentiality duties continue under the vendor agreement?",
        "The employee must report the suspected data breach to the Privacy Officer and Security team within 2 hours. A vendor that becomes aware of a personal data breach must notify NorthStar within 24 hours. Under the vendor agreement, confidentiality obligations survive termination for 5 years, or as long as information remains a trade secret.",
        ["Privacy Officer", "Security", "2 hours", "vendor", "24 hours", "confidentiality", "5 years", "trade secret"],
    ),
    # TEST: Hard, 2 sources - leave_policy + benefits_guide.
    (
        "A federally regulated NorthStar employee has completed 30 days of service, needs paid medical leave, has two children under age 7, and wants to take a role-relevant course. What paid leave, childcare support, and learning allowance are available?",
        "After 30 days of continuous employment, the employee earns 3 paid medical-leave days and can earn up to 10 days per calendar year. They may receive a CAD 150 monthly childcare subsidy for each child under age 7, up to 2 children. They also have a CAD 2,000 annual learning budget for a role-relevant course, with receipts submitted within 30 days.",
        ["30 days", "3 paid medical-leave days", "10 days", "CAD 150", "2 children", "CAD 2,000", "learning budget", "receipts"],
    ),
]


def load_ground_truth() -> list[tuple[str, str, list[str]]]:
    """Return list of (question, expected_answer, keywords) tuples."""
    return TEST_QUERIES


def get_query(index: int = 0) -> tuple[str, str, list[str]]:
    """Get a specific test query by index."""
    return TEST_QUERIES[index]
