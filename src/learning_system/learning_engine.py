"""
Learning Engine for Intelligent Document Q&A System
Analyzes feedback patterns and generates adaptive improvements
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import re
import asyncio
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from loguru import logger

from .feedback_collector import UserFeedback, FeedbackType, LearningPattern, feedback_collector


@dataclass
class AdaptationRule:
    """Rules for adapting Q&A responses based on learned patterns"""
    rule_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    rule_name: str = ""
    condition: str = ""  # When to apply this rule
    adaptation: str = ""  # How to modify the response
    confidence: float = 0.0
    effectiveness_score: float = 0.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LearningEngine:
    """Analyzes feedback and generates adaptive improvements"""
    
    def __init__(self, learning_directory: str = "data/learning"):
        self.learning_directory = Path(learning_directory)
        self.learning_directory.mkdir(parents=True, exist_ok=True)
        
        self.patterns_file = self.learning_directory / "learned_patterns.json"
        self.rules_file = self.learning_directory / "adaptation_rules.json"
        
        self.learned_patterns: List[LearningPattern] = []
        self.adaptation_rules: List[AdaptationRule] = []
        
        self._load_patterns()
        self._load_rules()
    
    def _load_patterns(self):
        """Load learned patterns from disk"""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.learned_patterns = []
                for item in data.get('patterns', []):
                    pattern = LearningPattern(
                        pattern_id=item.get('pattern_id', ''),
                        pattern_type=item.get('pattern_type', ''),
                        pattern_description=item.get('pattern_description', ''),
                        confidence_score=item.get('confidence_score', 0.0),
                        supporting_feedback_count=item.get('supporting_feedback_count', 0),
                        improvement_suggestion=item.get('improvement_suggestion', ''),
                        created_at=datetime.fromisoformat(item.get('created_at', datetime.now().isoformat())),
                        last_updated=datetime.fromisoformat(item.get('last_updated', datetime.now().isoformat())),
                        metadata=item.get('metadata', {})
                    )
                    self.learned_patterns.append(pattern)
                
                logger.info(f"Loaded {len(self.learned_patterns)} learned patterns")
                
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            self.learned_patterns = []
    
    def _save_patterns(self):
        """Save learned patterns to disk"""
        try:
            data = {
                'patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'pattern_type': p.pattern_type,
                        'pattern_description': p.pattern_description,
                        'confidence_score': p.confidence_score,
                        'supporting_feedback_count': p.supporting_feedback_count,
                        'improvement_suggestion': p.improvement_suggestion,
                        'created_at': p.created_at.isoformat(),
                        'last_updated': p.last_updated.isoformat(),
                        'metadata': p.metadata
                    }
                    for p in self.learned_patterns
                ],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def _load_rules(self):
        """Load adaptation rules from disk"""
        try:
            if self.rules_file.exists():
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.adaptation_rules = []
                for item in data.get('rules', []):
                    rule = AdaptationRule(
                        rule_id=item.get('rule_id', ''),
                        rule_name=item.get('rule_name', ''),
                        condition=item.get('condition', ''),
                        adaptation=item.get('adaptation', ''),
                        confidence=item.get('confidence', 0.0),
                        effectiveness_score=item.get('effectiveness_score', 0.0),
                        usage_count=item.get('usage_count', 0),
                        created_at=datetime.fromisoformat(item.get('created_at', datetime.now().isoformat())),
                        last_used=datetime.fromisoformat(item['last_used']) if item.get('last_used') else None,
                        metadata=item.get('metadata', {})
                    )
                    self.adaptation_rules.append(rule)
                
                logger.info(f"Loaded {len(self.adaptation_rules)} adaptation rules")
                
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            self.adaptation_rules = []
    
    def _save_rules(self):
        """Save adaptation rules to disk"""
        try:
            data = {
                'rules': [
                    {
                        'rule_id': r.rule_id,
                        'rule_name': r.rule_name,
                        'condition': r.condition,
                        'adaptation': r.adaptation,
                        'confidence': r.confidence,
                        'effectiveness_score': r.effectiveness_score,
                        'usage_count': r.usage_count,
                        'created_at': r.created_at.isoformat(),
                        'last_used': r.last_used.isoformat() if r.last_used else None,
                        'metadata': r.metadata
                    }
                    for r in self.adaptation_rules
                ],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.rules_file, 'w', encoding='utf-8') as f:                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving rules: {e}")
    
    async def analyze_feedback_patterns(self, min_pattern_support: int = 2) -> List[LearningPattern]:
        """Analyze feedback to identify patterns and learning opportunities"""
        try:
            feedback_history = feedback_collector.feedback_history
            new_patterns = []
            
            if len(feedback_history) < min_pattern_support:
                logger.info("Insufficient feedback for pattern analysis")
                return []
            
            # Pattern 1: Question type preferences
            question_patterns = await self._analyze_question_type_patterns(feedback_history)
            new_patterns.extend(question_patterns)
            
            # Pattern 2: Answer length preferences
            length_patterns = await self._analyze_answer_length_patterns(feedback_history)
            new_patterns.extend(length_patterns)
            
            # Pattern 3: Source citation preferences
            citation_patterns = await self._analyze_citation_patterns(feedback_history)
            new_patterns.extend(citation_patterns)
            
            # Pattern 4: Topic-specific preferences
            topic_patterns = await self._analyze_topic_patterns(feedback_history)
            new_patterns.extend(topic_patterns)
            
            # Pattern 5: User correction patterns
            correction_patterns = await self._analyze_correction_patterns(feedback_history)
            new_patterns.extend(correction_patterns)
              # Filter patterns by confidence and support
            validated_patterns = [p for p in new_patterns 
                                if p.confidence_score >= 0.4 and p.supporting_feedback_count >= min_pattern_support]
            
            # Update existing patterns or add new ones
            for pattern in validated_patterns:
                await self._update_or_add_pattern(pattern)
            
            self._save_patterns()
            logger.info(f"Analyzed feedback and found {len(validated_patterns)} significant patterns")
            
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {e}")
            return []
    
    async def _analyze_question_type_patterns(self, feedback_history: List[UserFeedback]) -> List[LearningPattern]:
        """Analyze patterns in question types and their feedback"""
        patterns = []
        
        # Group feedback by question type
        question_types = {
            'what': [],
            'how': [],
            'why': [],
            'when': [],
            'where': [],
            'summary': [],
            'details': [],
            'comparison': []
        }
        
        for feedback in feedback_history:
            question_lower = feedback.question.lower()
            
            if question_lower.startswith('what'):
                question_types['what'].append(feedback)
            elif question_lower.startswith('how'):
                question_types['how'].append(feedback)
            elif question_lower.startswith('why'):
                question_types['why'].append(feedback)
            elif question_lower.startswith('when'):
                question_types['when'].append(feedback)
            elif question_lower.startswith('where'):
                question_types['where'].append(feedback)
            elif any(word in question_lower for word in ['summary', 'summarize', 'overview']):
                question_types['summary'].append(feedback)
            elif any(word in question_lower for word in ['detail', 'specific', 'elaborate']):
                question_types['details'].append(feedback)
            elif any(word in question_lower for word in ['compare', 'difference', 'versus']):
                question_types['comparison'].append(feedback)
          # Analyze each question type
        for q_type, feedbacks in question_types.items():
            if len(feedbacks) >= 2:
                avg_rating = self._calculate_average_rating(feedbacks)
                
                if avg_rating < 3.0:  # Poor ratings
                    pattern = LearningPattern(
                        pattern_type="question_type_performance",
                        pattern_description=f"'{q_type}' questions receive lower satisfaction ratings",
                        confidence_score=min(1.0, len(feedbacks) / 10),
                        supporting_feedback_count=len(feedbacks),
                        improvement_suggestion=f"Improve response quality for '{q_type}' type questions",
                        metadata={"question_type": q_type, "avg_rating": avg_rating}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_answer_length_patterns(self, feedback_history: List[UserFeedback]) -> List[LearningPattern]:
        """Analyze preferences for answer length"""
        patterns = []
        
        # Group by answer length
        short_answers = []  # < 100 chars
        medium_answers = []  # 100-500 chars
        long_answers = []  # > 500 chars
        
        for feedback in feedback_history:
            answer_length = len(feedback.answer)
            
            if answer_length < 100:
                short_answers.append(feedback)
            elif answer_length <= 500:
                medium_answers.append(feedback)
            else:
                long_answers.append(feedback)
        
        # Analyze satisfaction by length
        length_categories = {
            "short": short_answers,
            "medium": medium_answers,
            "long": long_answers        }
        
        best_length = None
        best_rating = 0
        
        for length_type, feedbacks in length_categories.items():
            if len(feedbacks) >= 2:
                avg_rating = self._calculate_average_rating(feedbacks)
                
                if avg_rating > best_rating:
                    best_rating = avg_rating
                    best_length = length_type
        
        if best_length and best_rating > 3.5:
            pattern = LearningPattern(
                pattern_type="answer_length_preference",
                pattern_description=f"Users prefer {best_length} answers (avg rating: {best_rating:.2f})",
                confidence_score=min(1.0, sum(len(feedbacks) for feedbacks in length_categories.values()) / 20),
                supporting_feedback_count=sum(len(feedbacks) for feedbacks in length_categories.values()),
                improvement_suggestion=f"Optimize answer length to be {best_length}",
                metadata={"preferred_length": best_length, "avg_rating": best_rating}
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _analyze_citation_patterns(self, feedback_history: List[UserFeedback]) -> List[LearningPattern]:
        """Analyze patterns in source citation preferences"""
        patterns = []
        
        cited_answers = []
        uncited_answers = []
        
        for feedback in feedback_history:
            if '[Source:' in feedback.answer:
                cited_answers.append(feedback)
            else:
                uncited_answers.append(feedback)
        
        if len(cited_answers) >= 2 and len(uncited_answers) >= 2:
            cited_avg = self._calculate_average_rating(cited_answers)
            uncited_avg = self._calculate_average_rating(uncited_answers)
            
            if cited_avg > uncited_avg + 0.5:  # Significant preference for citations
                pattern = LearningPattern(
                    pattern_type="citation_preference",
                    pattern_description=f"Users prefer answers with source citations (cited: {cited_avg:.2f} vs uncited: {uncited_avg:.2f})",
                    confidence_score=min(1.0, (len(cited_answers) + len(uncited_answers)) / 15),
                    supporting_feedback_count=len(cited_answers) + len(uncited_answers),
                    improvement_suggestion="Always include source citations in answers",
                    metadata={"cited_rating": cited_avg, "uncited_rating": uncited_avg}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_topic_patterns(self, feedback_history: List[UserFeedback]) -> List[LearningPattern]:
        """Analyze patterns by topic/domain"""
        patterns = []
        
        # Simple topic extraction based on common keywords
        topic_groups = defaultdict(list)
        
        for feedback in feedback_history:
            question_words = set(feedback.question.lower().split())
            
            # Technology topics
            if any(word in question_words for word in ['technology', 'tech', 'software', 'hardware', 'computer']):
                topic_groups['technology'].append(feedback)
            
            # Business topics
            elif any(word in question_words for word in ['business', 'company', 'market', 'strategy', 'revenue']):
                topic_groups['business'].append(feedback)
            
            # Research topics
            elif any(word in question_words for word in ['research', 'study', 'analysis', 'data', 'findings']):
                topic_groups['research'].append(feedback)
            
            # General
            else:
                topic_groups['general'].append(feedback)
        
        # Analyze each topic group
        for topic, feedbacks in topic_groups.items():
            if len(feedbacks) >= 3:
                avg_rating = self._calculate_average_rating(feedbacks)
                
                if avg_rating < 3.0:  # Poor performance on this topic
                    pattern = LearningPattern(
                        pattern_type="topic_performance",
                        pattern_description=f"Lower satisfaction for {topic} questions (avg: {avg_rating:.2f})",
                        confidence_score=min(1.0, len(feedbacks) / 10),
                        supporting_feedback_count=len(feedbacks),
                        improvement_suggestion=f"Improve knowledge base and response quality for {topic} domain",
                        metadata={"topic": topic, "avg_rating": avg_rating}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_correction_patterns(self, feedback_history: List[UserFeedback]) -> List[LearningPattern]:
        """Analyze user correction patterns"""
        patterns = []
        
        corrections = [fb for fb in feedback_history if fb.user_correction]
        
        if len(corrections) >= 2:
            # Common correction types
            correction_types = Counter()
            
            for feedback in corrections:
                correction_lower = feedback.user_correction.lower()
                
                if any(word in correction_lower for word in ['more detail', 'elaborate', 'specific']):
                    correction_types['needs_more_detail'] += 1
                elif any(word in correction_lower for word in ['too long', 'concise', 'shorter']):
                    correction_types['too_verbose'] += 1
                elif any(word in correction_lower for word in ['incorrect', 'wrong', 'error']):
                    correction_types['factual_error'] += 1
                elif any(word in correction_lower for word in ['source', 'citation', 'reference']):
                    correction_types['missing_sources'] += 1
            
            # Create patterns for common correction types
            for correction_type, count in correction_types.items():
                if count >= 2:
                    pattern = LearningPattern(
                        pattern_type="correction_pattern",
                        pattern_description=f"Users frequently correct for {correction_type.replace('_', ' ')} ({count} times)",
                        confidence_score=min(1.0, count / 5),
                        supporting_feedback_count=count,
                        improvement_suggestion=f"Address {correction_type.replace('_', ' ')} in response generation",
                        metadata={"correction_type": correction_type, "frequency": count}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_average_rating(self, feedbacks: List[UserFeedback]) -> float:
        """Calculate average rating from feedback list"""
        ratings = []
        
        for feedback in feedbacks:
            if feedback.feedback_type == FeedbackType.RATING and isinstance(feedback.feedback_value, (int, float)):
                ratings.append(feedback.feedback_value)
            elif feedback.feedback_type == FeedbackType.THUMBS_UP:
                ratings.append(5)
            elif feedback.feedback_type == FeedbackType.THUMBS_DOWN:
                ratings.append(1)
        
        return sum(ratings) / len(ratings) if ratings else 2.5
    
    async def _update_or_add_pattern(self, new_pattern: LearningPattern):
        """Update existing pattern or add new one"""
        # Check if similar pattern exists
        for existing_pattern in self.learned_patterns:
            if (existing_pattern.pattern_type == new_pattern.pattern_type and 
                existing_pattern.metadata.get('question_type') == new_pattern.metadata.get('question_type') and
                existing_pattern.metadata.get('topic') == new_pattern.metadata.get('topic')):
                
                # Update existing pattern
                existing_pattern.confidence_score = max(existing_pattern.confidence_score, new_pattern.confidence_score)
                existing_pattern.supporting_feedback_count = new_pattern.supporting_feedback_count
                existing_pattern.last_updated = datetime.now()
                return
        
        # Add new pattern
        self.learned_patterns.append(new_pattern)
    
    async def generate_adaptation_rules(self) -> List[AdaptationRule]:
        """Generate adaptation rules from learned patterns"""
        try:
            new_rules = []
            
            for pattern in self.learned_patterns:
                if pattern.confidence_score >= 0.7:
                    rules = await self._pattern_to_rules(pattern)
                    new_rules.extend(rules)
            
            # Update existing rules or add new ones
            for rule in new_rules:
                await self._update_or_add_rule(rule)
            
            self._save_rules()
            logger.info(f"Generated {len(new_rules)} adaptation rules")
            
            return new_rules
            
        except Exception as e:
            logger.error(f"Error generating adaptation rules: {e}")
            return []
    
    async def _pattern_to_rules(self, pattern: LearningPattern) -> List[AdaptationRule]:
        """Convert a learning pattern into actionable adaptation rules"""
        rules = []
        
        if pattern.pattern_type == "answer_length_preference":
            preferred_length = pattern.metadata.get('preferred_length')
            
            if preferred_length == 'short':
                rule = AdaptationRule(
                    rule_name="Prefer Concise Answers",
                    condition="answer_length > 150",
                    adaptation="Make answer more concise, focus on key points",
                    confidence=pattern.confidence_score,
                    metadata={"pattern_id": pattern.pattern_id}
                )
                rules.append(rule)
            
            elif preferred_length == 'long':
                rule = AdaptationRule(
                    rule_name="Provide Detailed Answers",
                    condition="answer_length < 200",
                    adaptation="Expand answer with more details and examples",
                    confidence=pattern.confidence_score,
                    metadata={"pattern_id": pattern.pattern_id}
                )
                rules.append(rule)
        
        elif pattern.pattern_type == "citation_preference":
            rule = AdaptationRule(
                rule_name="Always Include Citations",
                condition="no_citations_in_answer",
                adaptation="Add [Source: ...] citations to support claims",
                confidence=pattern.confidence_score,
                metadata={"pattern_id": pattern.pattern_id}
            )
            rules.append(rule)
        
        elif pattern.pattern_type == "question_type_performance":
            question_type = pattern.metadata.get('question_type')
            rule = AdaptationRule(
                rule_name=f"Improve {question_type.title()} Questions",
                condition=f"question_starts_with_{question_type}",
                adaptation=f"Use enhanced response template for {question_type} questions",
                confidence=pattern.confidence_score,
                metadata={"pattern_id": pattern.pattern_id, "question_type": question_type}
            )
            rules.append(rule)
        
        return rules
    
    async def _update_or_add_rule(self, new_rule: AdaptationRule):
        """Update existing rule or add new one"""
        # Check if similar rule exists
        for existing_rule in self.adaptation_rules:
            if existing_rule.rule_name == new_rule.rule_name:
                # Update existing rule
                existing_rule.confidence = max(existing_rule.confidence, new_rule.confidence)
                existing_rule.metadata.update(new_rule.metadata)
                return
        
        # Add new rule
        self.adaptation_rules.append(new_rule)
    
    def get_applicable_rules(self, question: str, answer: str) -> List[AdaptationRule]:
        """Get adaptation rules that apply to the current question/answer"""
        applicable_rules = []
        
        for rule in self.adaptation_rules:
            if self._rule_applies(rule, question, answer):
                applicable_rules.append(rule)
        
        # Sort by confidence and effectiveness
        applicable_rules.sort(key=lambda r: (r.confidence, r.effectiveness_score), reverse=True)
        return applicable_rules
    
    def _rule_applies(self, rule: AdaptationRule, question: str, answer: str) -> bool:
        """Check if a rule applies to the current context"""
        condition = rule.condition.lower()
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        if "answer_length >" in condition:
            threshold = int(condition.split(">")[1].strip())
            return len(answer) > threshold
        
        elif "answer_length <" in condition:
            threshold = int(condition.split("<")[1].strip())
            return len(answer) < threshold
        
        elif "no_citations_in_answer" in condition:
            return '[Source:' not in answer
        
        elif "question_starts_with_" in condition:
            question_type = condition.replace("question_starts_with_", "")
            return question_lower.startswith(question_type)
        
        return False
    
    def update_rule_effectiveness(self, rule_id: str, feedback_improvement: float):
        """Update rule effectiveness based on user feedback"""
        for rule in self.adaptation_rules:
            if rule.rule_id == rule_id:
                rule.usage_count += 1
                rule.last_used = datetime.now()
                
                # Update effectiveness score (moving average)
                alpha = 0.1  # Learning rate
                rule.effectiveness_score = (1 - alpha) * rule.effectiveness_score + alpha * feedback_improvement
                
                self._save_rules()
                break
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        feedback_stats = feedback_collector.get_feedback_statistics()
        
        pattern_types = Counter(p.pattern_type for p in self.learned_patterns)
        rule_types = Counter(r.rule_name for r in self.adaptation_rules)
        
        avg_pattern_confidence = np.mean([p.confidence_score for p in self.learned_patterns]) if self.learned_patterns else 0
        avg_rule_effectiveness = np.mean([r.effectiveness_score for r in self.adaptation_rules]) if self.adaptation_rules else 0
        
        return {
            "feedback_statistics": feedback_stats,
            "total_patterns": len(self.learned_patterns),
            "pattern_types": dict(pattern_types),
            "average_pattern_confidence": round(avg_pattern_confidence, 3),
            "total_rules": len(self.adaptation_rules),
            "rule_types": dict(rule_types),
            "average_rule_effectiveness": round(avg_rule_effectiveness, 3),
            "active_rules": len([r for r in self.adaptation_rules if r.last_used]),
            "last_analysis": max([p.last_updated for p in self.learned_patterns]).isoformat() if self.learned_patterns else None
        }
    
    def get_patterns(self) -> List[LearningPattern]:
        """Get all learned patterns"""
        return self.learned_patterns.copy()
    
    def get_adaptation_rules(self) -> List[AdaptationRule]:
        """Get all adaptation rules"""
        return self.adaptation_rules.copy()


# Global learning engine instance
learning_engine = LearningEngine()
