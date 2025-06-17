"""
Response Adaptation System for Intelligent Document Q&A
Applies learned patterns and rules to improve response generation
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import asyncio
from dataclasses import dataclass
from loguru import logger

from .learning_engine import learning_engine, AdaptationRule
from .feedback_collector import feedback_collector


@dataclass
class AdaptedResponse:
    """Response that has been adapted based on learned patterns"""
    original_response: Any  # QAResponse - using Any to avoid circular import
    adapted_answer: str
    adapted_confidence: float
    adaptations_applied: List[str]
    adaptation_reasoning: str
    improvement_score: float = 0.0


class ResponseAdapter:
    """Adapts Q&A responses based on learned patterns and rules"""
    
    def __init__(self):
        self.adaptation_history: List[Dict[str, Any]] = []
    
    async def adapt_response(self, question: str, original_response: Any, 
                           context: Dict[str, Any] = None) -> AdaptedResponse:
        """Adapt a Q&A response based on learned patterns"""
        try:
            # Get applicable adaptation rules
            applicable_rules = learning_engine.get_applicable_rules(question, original_response.answer)
            
            if not applicable_rules:
                # No adaptations needed
                return AdaptedResponse(
                    original_response=original_response,
                    adapted_answer=original_response.answer,
                    adapted_confidence=original_response.confidence_score,
                    adaptations_applied=[],
                    adaptation_reasoning="No applicable adaptation rules found"
                )
            
            # Apply adaptations in order of priority
            adapted_answer = original_response.answer
            adapted_confidence = original_response.confidence_score
            adaptations_applied = []
            reasoning_parts = []
            
            for rule in applicable_rules[:3]:  # Apply top 3 rules max
                adaptation_result = await self._apply_adaptation_rule(
                    rule, question, adapted_answer, context
                )
                
                if adaptation_result:
                    adapted_answer = adaptation_result['adapted_answer']
                    adapted_confidence = adaptation_result['adapted_confidence']
                    adaptations_applied.append(rule.rule_name)
                    reasoning_parts.append(adaptation_result['reasoning'])
                    
                    # Update rule usage
                    learning_engine.update_rule_effectiveness(rule.rule_id, 0.5)  # Neutral assumption
            
            adaptation_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No adaptations applied"
            
            # Calculate improvement score (simplified)
            improvement_score = self._calculate_improvement_score(
                original_response.answer, adapted_answer, adaptations_applied
            )
            
            adapted_response = AdaptedResponse(
                original_response=original_response,
                adapted_answer=adapted_answer,
                adapted_confidence=adapted_confidence,
                adaptations_applied=adaptations_applied,
                adaptation_reasoning=adaptation_reasoning,
                improvement_score=improvement_score
            )
            
            # Track adaptation
            self._track_adaptation(question, original_response, adapted_response)
            
            logger.info(f"Applied {len(adaptations_applied)} adaptations to response")
            return adapted_response
            
        except Exception as e:
            logger.error(f"Error adapting response: {e}")
            return AdaptedResponse(
                original_response=original_response,
                adapted_answer=original_response.answer,
                adapted_confidence=original_response.confidence_score,
                adaptations_applied=[],
                adaptation_reasoning=f"Adaptation failed: {str(e)}"
            )
    
    async def _apply_adaptation_rule(self, rule: AdaptationRule, question: str, 
                                   answer: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Apply a specific adaptation rule to the answer"""
        try:
            adapted_answer = answer
            adapted_confidence = context.get('confidence', 0.8) if context else 0.8
            reasoning = ""
            
            if rule.rule_name == "Prefer Concise Answers":
                adapted_answer, reasoning = await self._make_concise(answer)
                adapted_confidence *= 0.95  # Slight confidence adjustment
            
            elif rule.rule_name == "Provide Detailed Answers":
                adapted_answer, reasoning = await self._add_details(answer, question, context)
                adapted_confidence *= 1.05  # Boost confidence for detailed answers
            
            elif rule.rule_name == "Always Include Citations":
                adapted_answer, reasoning = await self._add_citations(answer, context)
                adapted_confidence *= 1.1  # Boost confidence for cited answers
            
            elif "Improve" in rule.rule_name and "Questions" in rule.rule_name:
                adapted_answer, reasoning = await self._improve_question_type(answer, question, rule)
                adapted_confidence *= 1.02
            
            else:
                # Generic rule application
                adapted_answer, reasoning = await self._apply_generic_rule(rule, answer, question)
            
            if adapted_answer != answer:
                return {
                    'adapted_answer': adapted_answer,
                    'adapted_confidence': min(1.0, max(0.0, adapted_confidence)),
                    'reasoning': reasoning
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error applying rule {rule.rule_name}: {e}")
            return None
    
    async def _make_concise(self, answer: str) -> Tuple[str, str]:
        """Make answer more concise"""
        if len(answer) <= 150:
            return answer, "Answer already concise"
        
        # Simple conciseness strategies
        sentences = answer.split('. ')
        
        # Keep the most important sentences (first and those with key information)
        important_sentences = []
        
        # Always keep the first sentence
        if sentences:
            important_sentences.append(sentences[0])
        
        # Add sentences with key indicators
        for sentence in sentences[1:]:
            if any(word in sentence.lower() for word in ['important', 'key', 'main', 'primary', 'essential']):
                important_sentences.append(sentence)
            elif len(important_sentences) < 2:  # Keep at least 2 sentences
                important_sentences.append(sentence)
        
        concise_answer = '. '.join(important_sentences)
        if not concise_answer.endswith('.'):
            concise_answer += '.'
        
        reasoning = f"Condensed from {len(sentences)} to {len(important_sentences)} sentences"
        return concise_answer, reasoning
    
    async def _add_details(self, answer: str, question: str, context: Dict[str, Any] = None) -> Tuple[str, str]:
        """Add more details to the answer"""
        if len(answer) >= 400:
            return answer, "Answer already detailed"
        
        # Strategy: Add contextual information and elaboration
        detailed_answer = answer
        
        # Add context from sources if available
        sources = context.get('sources', []) if context else []
        if sources and len(sources) > 0:
            # Add a brief elaboration
            detailed_answer += f"\n\nBased on the available sources, this information is supported by {len(sources)} document(s) in the knowledge base."
        
        # Add clarifying information based on question type
        question_lower = question.lower()
        if question_lower.startswith('what'):
            detailed_answer += " This provides a comprehensive overview of the topic in question."
        elif question_lower.startswith('how'):
            detailed_answer += " Following these steps systematically will help achieve the desired outcome."
        elif question_lower.startswith('why'):
            detailed_answer += " Understanding these underlying reasons is crucial for a complete picture."
        
        reasoning = f"Expanded answer from {len(answer)} to {len(detailed_answer)} characters"
        return detailed_answer, reasoning
    
    async def _add_citations(self, answer: str, context: Dict[str, Any] = None) -> Tuple[str, str]:
        """Add source citations to the answer"""
        if '[Source:' in answer:
            return answer, "Citations already present"
        
        sources = context.get('sources', []) if context else []
        if not sources:
            # Add a generic citation disclaimer
            cited_answer = answer + " [Source: Knowledge Base]"
            reasoning = "Added generic knowledge base citation"
        else:
            # Add specific source citations
            source_list = [src.get('source', 'Unknown') for src in sources[:3]]
            citation = f" [Sources: {', '.join(source_list)}]"
            cited_answer = answer + citation
            reasoning = f"Added citations for {len(source_list)} sources"
        
        return cited_answer, reasoning
    
    async def _improve_question_type(self, answer: str, question: str, rule: AdaptationRule) -> Tuple[str, str]:
        """Improve answer based on question type patterns"""
        question_type = rule.metadata.get('question_type', 'general')
        
        if question_type == 'what':
            # What questions benefit from clear definitions and examples
            improved_answer = f"To answer your question directly: {answer}"
            if "example" not in answer.lower():
                improved_answer += " For example, this could be applied in various contexts to achieve similar results."
        
        elif question_type == 'how':
            # How questions benefit from step-by-step structure
            if "step" not in answer.lower():
                improved_answer = f"Here's how to approach this: {answer} This can be broken down into manageable steps for easier implementation."
            else:
                improved_answer = answer
        
        elif question_type == 'summary':
            # Summary questions benefit from structured organization
            improved_answer = f"Summary: {answer}"
            if not any(word in answer.lower() for word in ['first', 'second', 'finally', 'key points']):
                improved_answer += " These key points provide a comprehensive overview of the main topics."
        
        else:
            improved_answer = answer
        
        reasoning = f"Applied {question_type} question optimization"
        return improved_answer, reasoning
    
    async def _apply_generic_rule(self, rule: AdaptationRule, answer: str, question: str) -> Tuple[str, str]:
        """Apply a generic adaptation rule"""
        # Basic implementation - could be enhanced with more sophisticated NLP
        adaptation = rule.adaptation.lower()
        
        if "concise" in adaptation:
            return await self._make_concise(answer)
        elif "detail" in adaptation or "expand" in adaptation:
            return await self._add_details(answer, question)
        elif "citation" in adaptation or "source" in adaptation:
            return await self._add_citations(answer)
        else:
            # Generic improvement
            improved_answer = answer + f" [Improved per rule: {rule.rule_name}]"
            reasoning = f"Applied generic rule: {rule.rule_name}"
            return improved_answer, reasoning
    
    def _calculate_improvement_score(self, original_answer: str, adapted_answer: str, 
                                   adaptations: List[str]) -> float:
        """Calculate an improvement score for the adaptation"""
        score = 0.0
        
        # Length improvement
        length_ratio = len(adapted_answer) / len(original_answer) if original_answer else 1.0
        if 0.8 <= length_ratio <= 1.5:  # Good length ratio
            score += 0.2
        
        # Citation improvement
        if '[Source:' in adapted_answer and '[Source:' not in original_answer:
            score += 0.3
        
        # Structure improvement
        if adapted_answer.count('.') > original_answer.count('.'):
            score += 0.1
          # Number of adaptations
        score += min(0.4, len(adaptations) * 0.1)
        
        return min(1.0, score)
    
    def _track_adaptation(self, question: str, original_response: Any, 
                         adapted_response: AdaptedResponse):
        """Track adaptation for analysis"""
        adaptation_record = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'original_length': len(original_response.answer),
            'adapted_length': len(adapted_response.adapted_answer),
            'adaptations_applied': adapted_response.adaptations_applied,
            'improvement_score': adapted_response.improvement_score,
            'confidence_change': adapted_response.adapted_confidence - original_response.confidence_score
        }
        
        self.adaptation_history.append(adaptation_record)
          # Keep only recent history
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-1000:]
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about adaptations performed"""
        if not self.adaptation_history:
            return {
                "total_adaptations": 0,
                "average_improvement_score": 0.0,
                "adaptation_types": {},
                "average_length_change": 0.0,
                "average_confidence_change": 0.0,
                "successful_adaptations": 0,
                "recent_adaptations": 0
            }
        
        total_adaptations = len(self.adaptation_history)
        avg_improvement = sum(record['improvement_score'] for record in self.adaptation_history) / total_adaptations
        
        # Count adaptation types
        adaptation_counts = {}
        for record in self.adaptation_history:
            for adaptation in record['adaptations_applied']:
                adaptation_counts[adaptation] = adaptation_counts.get(adaptation, 0) + 1
        
        # Calculate average changes
        avg_length_change = sum(record['adapted_length'] - record['original_length'] 
                              for record in self.adaptation_history) / total_adaptations
        
        avg_confidence_change = sum(record['confidence_change'] 
                                  for record in self.adaptation_history) / total_adaptations
        
        return {
            "total_adaptations": total_adaptations,
            "average_improvement_score": round(avg_improvement, 3),
            "adaptation_types": adaptation_counts,
            "average_length_change": round(avg_length_change, 1),
            "average_confidence_change": round(avg_confidence_change, 3),
            "successful_adaptations": len([r for r in self.adaptation_history if r['improvement_score'] > 0.3]),
            "recent_adaptations": len([r for r in self.adaptation_history 
                                     if datetime.fromisoformat(r['timestamp']) > 
                                     datetime.now() - __import__('datetime').timedelta(days=7)])
        }


# Global response adapter instance
response_adapter = ResponseAdapter()
