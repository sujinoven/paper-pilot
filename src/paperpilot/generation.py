
from ollama import Client

from paperpilot.context import format_context
from paperpilot.citations import (
    validate_citations,
    check_citation_coverage,
)


SYSTEM_PROMPT = """
You are PaperPilot, a research-paper question-answering assistant.

EVIDENCE RULES:
1. Answer using only the provided research-paper passages.
2. Do not introduce facts, benefits, explanations, or motivations
   that are not supported by the passages.
3. Cite factual claims using the passage identifiers supplied
   in the context, such as [1], [2], or [3].
4. Bibliography references inside passages are not PaperPilot
   source citations. Never copy those reference numbers.
5. Never invent citations, equations, or unsupported details.

ANSWER QUALITY RULES:
6. Start with a direct answer to the user's question.
7. Explain the relevant mechanism or reason clearly and concisely.
8. Do not simply repeat the question as an explanation.
9. Do not add generic benefits such as improved performance,
   reduced overfitting, or better learning unless the passages
   explicitly support those claims.
10. Distinguish between how a mechanism works and why it is used.
    If the passages describe only how it works, do not invent
    a reason for its use.When asked why a mechanism is used, explain only motivations
explicitly supported by the passages. If the passages describe
the mechanism but do not explain its motivation, describe
the mechanism and state that the reason is not specified
in the provided evidence.
11. Keep related concepts separate. For example, do not confuse
    the motivation for multi-head attention with the motivation
    for scaling dot-product attention.
12. Prefer a short, well-supported answer over a longer answer
    containing speculative details.

MATHEMATICAL NOTATION:
13. If mathematical notation is unclear or corrupted in the
    extracted passages, explain the concept in words rather
    than reconstructing an equation.

INSUFFICIENT EVIDENCE:
14. If the passages do not provide enough evidence to answer
    the question, respond with exactly:
I could not find enough information in the provided paper passages to answer this question.

OUTPUT:
15. Return only the answer with appropriate source citations.
""".strip()

class PaperAnswerGenerator:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(host=host)

    def repair_citations(
        self,
        question: str,
        answer: str,
        results: list[dict],
    ) -> str:
        """
        Attempt to repair missing or invalid citations once.

        Uses the original retrieved passages as evidence.
        """
        context = format_context(results)

        repair_prompt = f"""
Research-paper evidence:

{context}

Question:
{question}

Previous answer:
{answer}

Revise the previous answer to use valid citations.

Rules:
1. Use only the evidence above.
2. Cite factual claims using passage identifiers such as [1].
3. Do not copy bibliography reference numbers.
4. Do not introduce new factual claims.
5. Do not invent or reconstruct mathematical equations.
6. If the evidence is insufficient, use the standard refusal.
7. Return only the revised answer.
""".strip()

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": repair_prompt,
                },
            ],
            options={
                "temperature": 0.0,
                "num_predict": 350,
            },
        )

        return response.message.content or ""

    def generate(
        self,
        question: str,
        results: list[dict],
    ) -> str:
        """
        Generate a grounded answer from retrieved passages.

        If citation validation fails, attempt one repair.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not results:
            return (
                "I could not find enough information in the "
                "provided paper passages to answer this question."
            )

        context = format_context(results)

        user_prompt = (
            f"Research-paper evidence:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer the question using only the evidence above. "
            "Include citations for factual claims."
        )

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0.1,
                "num_predict": 350,
            },
        )

        answer = response.message.content or ""

        validation = validate_citations(
            answer=answer,
            source_count=len(results),
        )

        coverage = check_citation_coverage(
            answer=answer,
            source_count=len(results),
        )

        if not coverage["coverage_ok"]:
            print(
                "WARNING: Citation check failed. "
                "Attempting repair."
            )

            repaired_answer = self.repair_citations(
                question=question,
                answer=answer,
                results=results,
            )

            repaired_validation = validate_citations(
                answer=repaired_answer,
                source_count=len(results),
            )

            repaired_coverage = check_citation_coverage(
                answer=repaired_answer,
                source_count=len(results),
            )

            if repaired_coverage["coverage_ok"]:
                answer = repaired_answer
                print("Citation repair passed validation.")
            else:
                print(
                    "WARNING: Citation repair failed.",
                    "Invalid citations:",
                    repaired_validation["invalid"],
                )

        return answer
