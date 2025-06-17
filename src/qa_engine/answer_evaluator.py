"""
Answer Quality Evaluation System
Evaluates Q&A responses for quality, accuracy, and relevance
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re

import google.generativeai as genai

from ..config import settings
from .memory_manager import ConversationTurn
from .qa_engine import QAResponse
from loguru import logger


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating answer quality"""
    relevance_score: float  # How relevant the answer is to the question (0-1)
    accuracy_score: float  # How accurate the answer appears to be (0-1)
    completeness_score: float  # How complete the answer is (0-1)
    clarity_score: float  # How clear and understandable the answer is (0-1)
    source_attribution_score: float  # How well sources are cited (0-1)
    confidence_alignment_score: float  # How well confidence aligns with quality (0-1)
    overall_score: float  # Weighted average of all scores (0-1)
    feedback_summary: str  # Textual feedback
    improvement_suggestions: List[str]  # Specific suggestions


@dataclass
class EvaluationResult:
    """Result of answer evaluation"""
    session_id: str
    turn_id: str
    question: str
    answer: str
    metrics: EvaluationMetrics
    evaluation_timestamp: datetime
    evaluator_version: str


class AnswerEvaluator:
    """Evaluates the quality of Q&A responses"""
    
    def __init__(self):
        self.model = None
        self.evaluation_history: List[EvaluationResult] = []
        self._initialize_model()
        
        # Evaluation weights
        self.metric_weights = {
            "relevance": 0.25,
            "accuracy": 0.25,
            "completeness": 0.20,
            "clarity": 0.15,
            "source_attribution": 0.10,
            "confidence_alignment": 0.05
        }
    
    def _initialize_model(self):
        """Initialize the evaluation model"""
        try:
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.2,  # Low temperature for consistent evaluations
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                    "response_mime_type": "text/plain",
                }
            )
            logger.info("Initialized evaluation model")
            
        except Exception as e:
            logger.error(f"Error initializing evaluation model: {e}")
            raise
    
    async def evaluate_response(self, qa_response: QAResponse, question: str) -> EvaluationResult:
        """Evaluate a Q&A response comprehensively"""
        try:
            # Run different evaluation metrics
            relevance = await self._evaluate_relevance(question, qa_response.answer)
            accuracy = await self._evaluate_accuracy(question, qa_response.answer, qa_response.context_used)
            completeness = await self._evaluate_completeness(question, qa_response.answer)
            clarity = self._evaluate_clarity(qa_response.answer)
            source_attribution = self._evaluate_source_attribution(qa_response.answer, qa_response.sources)
            confidence_alignment = self._evaluate_confidence_alignment(qa_response)
            
            # Generate improvement suggestions
            suggestions = await self._generate_improvement_suggestions(
                question, qa_response.answer, qa_response.context_used
            )
            
            # Calculate overall score
            overall_score = (
                relevance * self.metric_weights["relevance"] +
                accuracy * self.metric_weights["accuracy"] +
                completeness * self.metric_weights["completeness"] +
                clarity * self.metric_weights["clarity"] +
                source_attribution * self.metric_weights["source_attribution"] +
                confidence_alignment * self.metric_weights["confidence_alignment"]
            )
            
            # Generate feedback summary
            feedback = self._generate_feedback_summary(
                relevance, accuracy, completeness, clarity, source_attribution, confidence_alignment
            )
            
            # Create metrics object
            metrics = EvaluationMetrics(
                relevance_score=relevance,
                accuracy_score=accuracy,
                completeness_score=completeness,
                clarity_score=clarity,
                source_attribution_score=source_attribution,
                confidence_alignment_score=confidence_alignment,
                overall_score=overall_score,
                feedback_summary=feedback,
                improvement_suggestions=suggestions
            )
            
            # Create evaluation result
            result = EvaluationResult(
                session_id=qa_response.session_id,
                turn_id=qa_response.turn_id,
                question=question,
                answer=qa_response.answer,
                metrics=metrics,
                evaluation_timestamp=datetime.now(),
                evaluator_version="1.0"
            )
            
            # Store evaluation
            self.evaluation_history.append(result)
            
            logger.info(f"Evaluated response {qa_response.turn_id}: {overall_score:.2f} overall score")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            raise
    
    async def _evaluate_relevance(self, question: str, answer: str) -> float:
        """Evaluate how relevant the answer is to the question"""
        try:
            prompt = f"""
Evaluate how relevant this answer is to the given question on a scale of 0.0 to 1.0.

Question: {question}

Answer: {answer}

Consider:
- Does the answer address the main topic of the question?
- Does it answer what was specifically asked?
- Is the response on-topic?

Respond with only a number between 0.0 and 1.0, where:
0.0 = Completely irrelevant
0.5 = Somewhat relevant
1.0 = Perfectly relevant

Score:"""

            response = await asyncio.to_thread(self.model.generate_content, prompt)
            score_text = response.text.strip()
            
            # Extract score
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Could not parse relevance score: {score_text}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error evaluating relevance: {e}")
            return 0.5
    
    async def _evaluate_accuracy(self, question: str, answer: str, context_used: List[Dict[str, Any]]) -> float:
        """Evaluate the accuracy of the answer based on provided context"""
        try:
            context_text = "\\n\\n".join([
                f"Source: {ctx.get('source', 'Unknown')}\\nContent: {ctx.get('content', '')}"
                for ctx in context_used[:3]
            ])
            
            prompt = f"""
Evaluate how accurate this answer is based on the provided context on a scale of 0.0 to 1.0.

Question: {question}

Answer: {answer}

Context/Sources:
{context_text}

Consider:
- Does the answer accurately reflect the information in the context?
- Are there any factual errors or misrepresentations?
- Does the answer stay within the bounds of what the context supports?

Respond with only a number between 0.0 and 1.0, where:
0.0 = Completely inaccurate or unsupported
0.5 = Partially accurate
1.0 = Completely accurate and well-supported

Score:"""

            response = await asyncio.to_thread(self.model.generate_content, prompt)
            score_text = response.text.strip()
            
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Could not parse accuracy score: {score_text}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error evaluating accuracy: {e}")
            return 0.5
    
    async def _evaluate_completeness(self, question: str, answer: str) -> float:
        """Evaluate how complete the answer is"""
        try:
            prompt = f"""
Evaluate how complete this answer is for the given question on a scale of 0.0 to 1.0.

Question: {question}

Answer: {answer}

Consider:
- Does the answer address all parts of the question?
- Are important aspects or follow-up information included?
- Is the answer thorough enough for the complexity of the question?

Respond with only a number between 0.0 and 1.0, where:
0.0 = Very incomplete, major gaps
0.5 = Partially complete
1.0 = Very complete and comprehensive

Score:"""

            response = await asyncio.to_thread(self.model.generate_content, prompt)
            score_text = response.text.strip()
            
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Could not parse completeness score: {score_text}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error evaluating completeness: {e}")
            return 0.5
    
    def _evaluate_clarity(self, answer: str) -> float:
        """Evaluate the clarity and readability of the answer"""
        try:
            score = 0.5  # Base score
            
            # Check length (too short or too long reduces clarity)
            length = len(answer)
            if 50 <= length <= 500:
                score += 0.2
            elif length < 20 or length > 1000:
                score -= 0.2
            
            # Check for structure (paragraphs, bullet points)
            if '\\n' in answer or '•' in answer or '-' in answer:
                score += 0.1
            
            # Check for clear language (avoid too many complex words)
            words = answer.split()
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                if avg_word_length < 6:  # Reasonable average word length
                    score += 0.1
                else:
                    score -= 0.1
            
            # Check for hedging language that might confuse
            hedge_words = ['maybe', 'possibly', 'perhaps', 'might be', 'could be']
            hedge_count = sum(1 for hedge in hedge_words if hedge in answer.lower())
            if hedge_count > 2:
                score -= 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error evaluating clarity: {e}")
            return 0.5
    
    def _evaluate_source_attribution(self, answer: str, sources: List[str]) -> float:
        """Evaluate how well sources are cited in the answer"""
        try:
            score = 0.0
              # Check if sources are mentioned in the answer
            source_citations = len(re.findall(r'\[Source:', answer))
            
            if sources:
                if source_citations > 0:
                    score += 0.5  # Has some source attribution
                    
                    # Bonus for multiple citations
                    if source_citations >= len(sources):
                        score += 0.3  # Cites all or most sources
                    elif source_citations >= len(sources) // 2:
                        score += 0.2  # Cites at least half
                
                # Check for proper citation format
                if '[Source:' in answer and ']' in answer:
                    score += 0.2
            else:
                # If no sources available, check if answer acknowledges this
                if any(phrase in answer.lower() for phrase in ['based on', 'according to', 'from the document']):
                    score += 0.3
                else:
                    score = 0.7  # Not penalized heavily if no sources to cite
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error evaluating source attribution: {e}")
            return 0.5
    
    def _evaluate_confidence_alignment(self, qa_response: QAResponse) -> float:
        """Evaluate how well the confidence score aligns with answer quality"""
        try:
            # This is a meta-evaluation that compares stated confidence with apparent quality
            confidence = qa_response.confidence_score
            
            # Basic heuristics for answer quality indicators
            answer = qa_response.answer
            quality_indicators = 0
            
            # Positive indicators
            if len(answer) > 50:
                quality_indicators += 1
            if qa_response.sources:
                quality_indicators += 1
            if '[Source:' in answer:
                quality_indicators += 1
            if len(qa_response.context_used) > 0:
                quality_indicators += 1
                
            # Negative indicators
            uncertain_phrases = ['not sure', 'unclear', 'cannot determine', 'insufficient']
            if any(phrase in answer.lower() for phrase in uncertain_phrases):
                quality_indicators -= 2
            
            expected_confidence = min(1.0, max(0.0, 0.3 + (quality_indicators * 0.15)))
            
            # How well does actual confidence align with expected?
            alignment = 1.0 - abs(confidence - expected_confidence)
            
            return max(0.0, min(1.0, alignment))
            
        except Exception as e:
            logger.error(f"Error evaluating confidence alignment: {e}")
            return 0.5
    
    async def _generate_improvement_suggestions(self, question: str, answer: str, 
                                             context_used: List[Dict[str, Any]]) -> List[str]:
        """Generate specific suggestions for improving the answer"""
        try:
            suggestions = []
            
            # Length-based suggestions
            if len(answer) < 30:
                suggestions.append("Consider providing a more detailed explanation")
            elif len(answer) > 800:
                suggestions.append("Consider making the answer more concise")
              # Source citation suggestions
            if not re.search(r'\[Source:', answer) and context_used:
                suggestions.append("Add source citations to support claims")
            
            # Structure suggestions
            if len(answer) > 200 and '\\n' not in answer:
                suggestions.append("Break down the answer into paragraphs for better readability")
            
            # Certainty suggestions
            uncertain_count = sum(1 for phrase in ['might', 'maybe', 'possibly', 'unclear'] 
                                if phrase in answer.lower())
            if uncertain_count > 2:
                suggestions.append("Reduce uncertain language where possible")
            
            # Context utilization suggestions
            if len(context_used) > 0 and len(answer) < 100:
                suggestions.append("Utilize more information from the available context")
            
            return suggestions[:4]  # Limit to top 4 suggestions
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            return ["Consider reviewing and refining the answer"]
    
    def _generate_feedback_summary(self, relevance: float, accuracy: float, completeness: float,
                                 clarity: float, source_attribution: float, confidence_alignment: float) -> str:
        """Generate a human-readable feedback summary"""
        try:
            feedback_parts = []
            
            # Overall assessment
            overall = (relevance + accuracy + completeness + clarity + source_attribution + confidence_alignment) / 6
            
            if overall >= 0.8:
                feedback_parts.append("Excellent response overall.")
            elif overall >= 0.6:
                feedback_parts.append("Good response with room for improvement.")
            else:
                feedback_parts.append("Response needs significant improvement.")
            
            # Specific feedback
            if relevance < 0.6:
                feedback_parts.append("Answer should be more directly relevant to the question.")
            
            if accuracy < 0.6:
                feedback_parts.append("Accuracy could be improved by better following the source material.")
            
            if completeness < 0.6:
                feedback_parts.append("Answer could be more comprehensive.")
            
            if clarity < 0.6:
                feedback_parts.append("Clarity and readability could be enhanced.")
            
            if source_attribution < 0.6:
                feedback_parts.append("Better source citation would strengthen the response.")
            
            return " ".join(feedback_parts)
            
        except Exception as e:
            logger.error(f"Error generating feedback summary: {e}")
            return "Evaluation completed."
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get statistics about evaluations performed"""
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        scores = [eval_result.metrics.overall_score for eval_result in self.evaluation_history]
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "average_score": sum(scores) / len(scores),
            "score_distribution": {
                "excellent (0.8+)": len([s for s in scores if s >= 0.8]),
                "good (0.6-0.8)": len([s for s in scores if 0.6 <= s < 0.8]),
                "needs_improvement (<0.6)": len([s for s in scores if s < 0.6])
            },
            "latest_evaluations": [
                {
                    "session_id": result.session_id,
                    "score": result.metrics.overall_score,
                    "timestamp": result.evaluation_timestamp.isoformat()
                }
                for result in self.evaluation_history[-5:]
            ]
        }


# Global evaluator instance
answer_evaluator = AnswerEvaluator()
