# ENGINEERING_COMMUNICATION_RULES.md

This repository prioritizes technical correctness, reproducibility, and validation rigor over conversational style or optimistic framing.

Communication rules for agents and contributors:

- Use dry, technical, engineering-oriented language.
- Prefer precise statements over vague optimism.
- Avoid hype, motivational wording, emotional framing, or conversational filler.
- Avoid slang, memes, swearing, sarcasm, or exaggerated language.
- Do not use “AI assistant” style phrasing.
- Keep responses concise and implementation-focused.

Reasoning and reporting rules:

- Clearly separate facts, assumptions, hypotheses, and speculation.
- Explicitly identify uncertainty, limitations, and validation gaps.
- Do not overclaim based on small-sample experiments.
- Do not describe incomplete systems as production-ready.
- Treat smoke tests as smoke tests, not evidence of robustness.
- Treat leaderboard speculation as low-priority.

Engineering priorities:

1. Validation trustworthiness.
2. Reproducibility.
3. Generalization stability.
4. Experimental traceability.
5. OCR quality.
6. Inference robustness.
7. Training speed.
8. Code elegance.

Implementation principles:

- Prefer deterministic and reproducible pipelines.
- Avoid hidden state and notebook-driven workflows.
- Keep configurations explicit and versioned.
- Log important experiment metadata and limitations.
- Prioritize interpretable debugging and measurable improvements.
- Avoid premature optimization and overengineering.

Code review expectations:

- Identify leakage risks explicitly.
- Identify validation weaknesses explicitly.
- Explain trade-offs and failure modes.
- Prefer conservative conclusions over optimistic interpretation.
- Report confidence level when evidence is limited.

Language preference:

- Prefer English for technical discussions, code comments, experiment descriptions, and architecture decisions.
- Use consistent technical terminology.