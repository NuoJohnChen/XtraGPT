"""
Tests for paper_revision_bench.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from paper_revision_bench import (
    evaluate,
    evaluate_async,
    list_sections,
    list_criteria,
    get_criterion_prompt,
)
from paper_revision_bench.results import EvaluationResult, SampleResult, Winner
from paper_revision_bench.judges import create_judge, OpenAIJudge, AnthropicJudge, OllamaJudge


class TestCriteria:
    """Tests for criteria module."""

    def test_list_sections(self):
        sections = list_sections()
        assert isinstance(sections, list)
        assert "abstract" in sections
        assert "introduction" in sections
        assert "title" in sections
        assert len(sections) == 6

    def test_list_criteria_all(self):
        criteria = list_criteria()
        assert isinstance(criteria, list)
        assert "overall" in criteria
        assert "conciseness" in criteria
        assert "clarity" in criteria

    def test_list_criteria_for_section(self):
        criteria = list_criteria("abstract")
        assert isinstance(criteria, list)
        assert "overall" in criteria
        assert "conciseness" in criteria

    def test_list_criteria_invalid_section(self):
        with pytest.raises(ValueError):
            list_criteria("invalid_section")

    def test_get_criterion_prompt(self):
        prompt = get_criterion_prompt("abstract", "conciseness")
        assert isinstance(prompt, str)
        assert "ORIGINAL" in prompt
        assert "REVISED" in prompt
        assert "conciseness" in prompt.lower()


class TestResults:
    """Tests for results module."""

    def test_sample_result(self):
        result = SampleResult(
            index=0,
            winner=Winner.REVISED,
            score=0.8,
            explanation="Revised is better",
            original_text="original",
            revised_text="revised",
        )
        assert result.winner == Winner.REVISED
        assert result.score == 0.8

    def test_sample_result_to_dict(self):
        result = SampleResult(
            index=0,
            winner=Winner.REVISED,
            score=0.8,
            explanation="Test",
        )
        d = result.to_dict()
        assert d["winner"] == "revised"
        assert d["score"] == 0.8

    def test_evaluation_result_win_rate(self):
        details = [
            SampleResult(index=0, winner=Winner.REVISED, score=1.0, explanation=""),
            SampleResult(index=1, winner=Winner.REVISED, score=1.0, explanation=""),
            SampleResult(index=2, winner=Winner.ORIGINAL, score=0.0, explanation=""),
            SampleResult(index=3, winner=Winner.TIE, score=0.5, explanation=""),
        ]
        result = EvaluationResult(details=details)

        assert result.win_rate == 0.5  # 2/4
        assert result.lose_rate == 0.25  # 1/4
        assert result.tie_rate == 0.25  # 1/4
        assert result.n_wins == 2
        assert result.n_losses == 1
        assert result.n_ties == 1
        assert result.total == 4

    def test_evaluation_result_empty(self):
        result = EvaluationResult(details=[])
        assert result.win_rate == 0.0
        assert result.lose_rate == 0.0
        assert result.tie_rate == 0.0

    def test_evaluation_result_summary(self):
        details = [
            SampleResult(index=0, winner=Winner.REVISED, score=1.0, explanation=""),
        ]
        result = EvaluationResult(
            details=details,
            metadata={"judge_model": "gpt-4", "section": "abstract", "criterion": "overall"}
        )
        summary = result.summary()
        assert "Win Rate" in summary
        assert "gpt-4" in summary


class TestJudges:
    """Tests for judges module."""

    def test_create_judge_openai(self):
        judge = create_judge("gpt-4-turbo", api_key="test-key")
        assert isinstance(judge, OpenAIJudge)
        assert judge.model == "gpt-4-turbo"

    def test_create_judge_anthropic(self):
        judge = create_judge("claude-3-opus", api_key="test-key")
        assert isinstance(judge, AnthropicJudge)

    def test_create_judge_ollama(self):
        judge = create_judge("ollama/llama3")
        assert isinstance(judge, OllamaJudge)
        assert judge.model == "llama3"

    def test_parse_response_valid_json(self):
        judge = OpenAIJudge("gpt-4", api_key="test")
        response = '{"winner": "revised", "score": 0.8, "explanation": "Better clarity"}'
        result = judge._parse_response(response)

        assert result["winner"] == Winner.REVISED
        assert result["score"] == 0.8
        assert "Better clarity" in result["explanation"]

    def test_parse_response_with_markdown(self):
        judge = OpenAIJudge("gpt-4", api_key="test")
        response = '''Here is my analysis:
```json
{"winner": "original", "score": 0.2, "explanation": "Original is clearer"}
```'''
        result = judge._parse_response(response)

        assert result["winner"] == Winner.ORIGINAL
        assert result["score"] == 0.2

    def test_parse_response_tie(self):
        judge = OpenAIJudge("gpt-4", api_key="test")
        response = '{"winner": "tie", "score": 0.5, "explanation": "Both are equal"}'
        result = judge._parse_response(response)

        assert result["winner"] == Winner.TIE
        assert result["score"] == 0.5


class TestEvaluate:
    """Tests for core evaluate functions."""

    def test_evaluate_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate(
                original_texts=["a", "b"],
                revised_texts=["c"],
                judge_model="gpt-4",
                api_key="test",
            )

    def test_evaluate_invalid_section(self):
        with pytest.raises(ValueError, match="Invalid section"):
            evaluate(
                original_texts=["a"],
                revised_texts=["b"],
                section="invalid",
                judge_model="gpt-4",
                api_key="test",
            )

    def test_evaluate_invalid_criterion(self):
        with pytest.raises(ValueError, match="Invalid criterion"):
            evaluate(
                original_texts=["a"],
                revised_texts=["b"],
                section="abstract",
                criterion="invalid_criterion",
                judge_model="gpt-4",
                api_key="test",
            )

    @pytest.mark.asyncio
    async def test_evaluate_async_mock(self):
        """Test evaluate_async with mocked judge."""
        mock_response = '{"winner": "revised", "score": 0.9, "explanation": "Much better"}'

        with patch("paper_revision_bench.judges.OpenAIJudge._call_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await evaluate_async(
                original_texts=["Original text here"],
                revised_texts=["Revised text here"],
                judge_model="gpt-4-turbo",
                api_key="test-key",
                show_progress=False,
            )

            assert isinstance(result, EvaluationResult)
            assert result.total == 1
            assert result.win_rate == 1.0
            assert result.details[0].winner == Winner.REVISED


# Integration test (requires API key)
@pytest.mark.skip(reason="Requires API key")
class TestIntegration:
    """Integration tests that require actual API calls."""

    def test_evaluate_openai(self):
        result = evaluate(
            original_texts=["The method is very good and works well."],
            revised_texts=["The method achieves state-of-the-art performance."],
            section="abstract",
            criterion="impact",
            judge_model="gpt-4-turbo",
        )
        assert isinstance(result, EvaluationResult)
        assert result.total == 1
