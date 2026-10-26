from ollama import Client
from paperpilot.citations import validate_citations

from paperpilot.context import format_context


SYSTEM_PROMPT = """
You are PaperPilot, a research-paper question-answering assistant.

Rules:
1. Answer using only the provided research-paper passages.
2. Do not introduce facts from outside the passages.
3. Cite factual claims using the passage identifiers supplied
   in the context, such as [1], [2], or [3].
4. The passages may contain bibliography references such as [9]
   or [27]. These are NOT PaperPilot source citations.
5. Never copy bibliography reference numbers into your answer.
   Cite the numbered retrieved passage that supports the claim.
6. If the passages do not provide enough evidence, say:
   "I could not find enough information in the provided paper
   passages to answer this question."
7. If mathematical notation is unclear or corrupted in the
   extracted passages, explain the concept in words instead
   of guessing or reconstructing an equation.
8. Do not invent citations, equations, or unsupported details.
9. Keep answers clear and concise.
""".strip()


class PaperAnswerGenerator:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(host=host)

    def generate(
        self,
        question: str,
        results: list[dict],
    ) -> str:
        """
        Generate an answer grounded in retrieved passages.
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

        if not validation["all_valid"]:
            print(
                "WARNING: Invalid citations detected:",
                validation["invalid"],
            )

        return answer