from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


@dataclass
class RagEvaluatorConf:
    llm_model: str = "gpt-oss:20b"
    relevance_weight: float = 1.0
    faithfulness_weight: float = 1.0
    coherence_weight: float = 0.5


class RagEvaluator:
    def __init__(self, conf: RagEvaluatorConf, logger: Logger):
        self.conf = conf
        self.logger = logger
        self.llm = OllamaLLM(model=self.conf.llm_model)

    def evaluate(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Evaluate the quality and relevance of the answer.

        Args:
            question: The original question.
            answer: The generated answer.
            context: The retrieved context documents.

        Returns:
            A dictionary with evaluation scores and explanations.
        """
        relevance = self._evaluate_metric("relevance", question, answer)
        faithfulness = self._evaluate_metric("faithfulness", question, answer, context)
        coherence = self._evaluate_metric("coherence", question, answer)

        overall_score = (
            self.conf.relevance_weight * relevance["score"]
            + self.conf.faithfulness_weight * faithfulness["score"]
            + self.conf.coherence_weight * coherence["score"]
        ) / (
            self.conf.relevance_weight
            + self.conf.faithfulness_weight
            + self.conf.coherence_weight
        )

        return {
            "relevance": relevance,
            "faithfulness": faithfulness,
            "coherence": coherence,
            "overall_score": overall_score,
        }

    def _evaluate_metric(
        self, metric: str, question: str, answer: str, context: str = ""
    ) -> Dict[str, Any]:
        """Evaluate a specific metric using the LLM."""
        if metric == "relevance":
            prompt_template = """Evaluate the relevance of the following answer
to the question on a scale of 1-5.

Question: {question}
Answer: {answer}

Provide a score (1-5) and a brief explanation."""
            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": question, "answer": answer})
        elif metric == "faithfulness":
            prompt_template = """Evaluate how faithfully the following answer
reflects the provided context on a scale of 1-5.

Question: {question}
Context: {context}
Answer: {answer}

Provide a score (1-5) and a brief explanation."""
            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke(
                {"question": question, "context": context, "answer": answer}
            )
        elif metric == "coherence":
            prompt_template = """Evaluate the coherence of the following answer
on a scale of 1-5.

Answer: {answer}

Provide a score (1-5) and a brief explanation."""
            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"answer": answer})
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Parse the response to extract score and explanation
        lines = response.strip().split("\n")
        score_line = next(
            (
                line
                for line in lines
                if "score" in line.lower() or any(char.isdigit() for char in line)
            ),
            lines[0],
        )
        score = self._extract_score(score_line)
        explanation = (
            " ".join(lines[1:]) if len(lines) > 1 else "No explanation provided."
        )

        return {"score": score, "explanation": explanation, "raw_response": response}

    def _extract_score(self, score_line: str) -> float:
        """Extract the score from the response line."""
        import re

        match = re.search(r"(\d+(\.\d+)?)", score_line)
        if match:
            return float(match.group(1))
        else:
            self.logger.warning(f"Could not extract score from: {score_line}")
            return 3.0  # Default neutral score
