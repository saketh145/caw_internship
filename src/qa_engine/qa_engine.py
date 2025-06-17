"""
Context-Aware Q&A Engine with Memory and Learning
Handles question answering using retrieved documents and conversation context
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

import google.generativeai as genai

from ..config import settings
from ..document_processor.vector_database import vector_db
from ..document_processor.embedding_generator import embedding_generator
from .memory_manager import conversation_memory, ConversationTurn
from .types import QAResponse
from ..learning_system import response_adapter, feedback_collector, learning_engine
from loguru import logger


class ContextAwareQAEngine:
    """Advanced Q&A engine with memory and context awareness"""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
        self.max_context_length = 8000  # Token limit for context
        self.min_confidence_threshold = 0.3
        
    def _initialize_model(self):
        """Initialize the Gemini model for text generation"""
        try:
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.1,  # Low temperature for factual responses
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                    "response_mime_type": "text/plain",
                }
            )
            logger.info("Initialized Gemini model for Q&A")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            raise
    
    async def answer_question(self, question: str, session_id: Optional[str] = None,
                            user_id: Optional[str] = None) -> QAResponse:
        """Generate an answer to a question using documents and conversation context"""
        start_time = datetime.now()
        
        try:
            # Create session if needed
            if not session_id:
                session_id = conversation_memory.create_session(user_id)
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = await self._retrieve_relevant_documents(question, session_id)
            
            # Step 2: Get conversation context
            conversation_context = conversation_memory.get_session_context(session_id, max_turns=3)
            
            # Step 3: Generate response using context
            answer, confidence = await self._generate_contextual_answer(
                question, retrieved_docs, conversation_context
            )
            
            # Step 4: Extract sources and metadata
            sources = self._extract_sources(retrieved_docs)
            context_used = self._format_context_for_storage(retrieved_docs)
            
            # Step 5: Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Step 6: Store in conversation memory
            turn_id = conversation_memory.add_turn(
                session_id=session_id,
                user_question=question,
                system_answer=answer,
                context_used=context_used,
                confidence_score=confidence,
                sources=sources,
                metadata={
                    "processing_time": processing_time,
                    "retrieval_count": len(retrieved_docs),
                    "model_used": "gemini-1.5-flash"
                }
            )
            
            # Step 7: Create response object
            response = QAResponse(
                answer=answer,
                confidence_score=confidence,
                sources=sources,
                context_used=context_used,
                processing_time=processing_time,
                session_id=session_id,
                turn_id=turn_id,
                metadata={
                    "document_count": len(retrieved_docs),
                    "conversation_turns": len(conversation_context)
                }
            )
            
            logger.info(f"Generated answer for session {session_id} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Return error response
            processing_time = (datetime.now() - start_time).total_seconds()
            return QAResponse(
                answer="I apologize, but I encountered an error while processing your question. Please try again.",
                confidence_score=0.0,
                sources=[],
                context_used=[],
                processing_time=processing_time,
                session_id=session_id or "error",
                turn_id="error",
                metadata={"error": str(e)}
            )
    
    async def _retrieve_relevant_documents(self, question: str, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the question"""
        try:            # Generate embedding for the question
            question_embedding_result = await embedding_generator.generate_query_embedding(question)
            
            # Search vector database
            search_results = await vector_db.similarity_search(
                query_embedding=question_embedding_result.embedding,  # Extract the embedding list
                n_results=5,
                filter_criteria=None  # Could filter by session or user preferences
            )
            
            # Also check conversation history for relevant context
            historical_context = conversation_memory.search_conversation_history(question, max_results=3)
            
            # Combine and format results
            retrieved_docs = []
              # Add document search results
            if search_results:
                # Handle both raw ChromaDB format and our formatted results
                if isinstance(search_results, list):
                    # Our formatted results
                    for result in search_results:                        retrieved_docs.append({
                            "content": result.get("content", ""),
                            "source": result.get("metadata", {}).get("source", "Unknown"),
                            "score": result.get("similarity_score", 0.0),
                            "metadata": result.get("metadata", {}),
                            "type": "document"
                        })
                elif isinstance(search_results, dict) and 'documents' in search_results:
                    # ChromaDB format
                    for i, doc in enumerate(search_results['documents'][0]):  # First query results
                        doc_metadata = search_results['metadatas'][0][i] if search_results['metadatas'] else {}
                        distance = search_results['distances'][0][i] if search_results['distances'] else 1.0
                        
                        retrieved_docs.append({
                            "content": doc,
                            "source": doc_metadata.get("source_file", doc_metadata.get("source", "Unknown")),
                            "score": 1.0 - distance,  # Convert distance to similarity score
                            "metadata": doc_metadata,
                            "type": "document"
                        })
            
            # Add historical conversation context
            for turn in historical_context:
                retrieved_docs.append({
                    "content": f"Previous Q: {turn.user_question}\\nPrevious A: {turn.system_answer}",
                    "source": "conversation_history",
                    "score": 0.7,  # Lower score for historical context
                    "metadata": {"timestamp": turn.timestamp.isoformat()},
                    "type": "conversation"
                })
            
            # Sort by score
            retrieved_docs.sort(key=lambda x: x["score"], reverse=True)
            
            return retrieved_docs[:8]  # Limit to top 8 results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def _generate_contextual_answer(self, question: str, retrieved_docs: List[Dict[str, Any]],
                                        conversation_context: List[ConversationTurn]) -> Tuple[str, float]:
        """Generate an answer using retrieved documents and conversation context"""
        try:
            # Build the prompt
            prompt = self._build_qa_prompt(question, retrieved_docs, conversation_context)
            
            # Generate response
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            
            # Extract answer and calculate confidence
            answer = response.text.strip()
            
            # Simple confidence calculation based on response characteristics
            confidence = self._calculate_confidence(answer, retrieved_docs, question)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I couldn't generate a proper answer to your question.", 0.0
    
    def _build_qa_prompt(self, question: str, retrieved_docs: List[Dict[str, Any]],
                        conversation_context: List[ConversationTurn]) -> str:
        """Build a comprehensive prompt for the Q&A model"""
        
        prompt_parts = [
            "You are an intelligent document Q&A assistant. Your task is to provide accurate, helpful answers based on the provided context.",
            "",
            "INSTRUCTIONS:",
            "1. Answer based primarily on the provided documents",
            "2. If the conversation history is relevant, incorporate it naturally",
            "3. If you can't find a complete answer in the context, say so clearly",
            "4. Cite sources when possible (use [Source: filename] format)",
            "5. Be concise but comprehensive",
            "6. If the question is ambiguous, ask for clarification",
            "",
        ]
        
        # Add conversation context if available
        if conversation_context:
            prompt_parts.extend([
                "CONVERSATION HISTORY:",
                "Here are the recent exchanges in this conversation:",
            ])
            
            for i, turn in enumerate(conversation_context[-3:], 1):  # Last 3 turns
                prompt_parts.extend([
                    f"Turn {i}:",
                    f"Human: {turn.user_question}",
                    f"Assistant: {turn.system_answer[:200]}{'...' if len(turn.system_answer) > 200 else ''}",
                    ""
                ])
        
        # Add retrieved documents
        if retrieved_docs:
            prompt_parts.extend([
                "RELEVANT DOCUMENTS:",
                "Here are the most relevant documents for answering the question:",
                ""
            ])
            
            for i, doc in enumerate(retrieved_docs[:5], 1):  # Top 5 documents
                source_info = f" [Source: {doc['source']}]" if doc['source'] != 'Unknown' else ""
                prompt_parts.extend([
                    f"Document {i}{source_info}:",
                    doc['content'][:800] + ("..." if len(doc['content']) > 800 else ""),
                    ""
                ])
        
        # Add the current question
        prompt_parts.extend([
            "CURRENT QUESTION:",
            question,
            "",
            "ANSWER:",
            "Based on the provided context and conversation history, here is my answer:",
        ])
        
        return "\\n".join(prompt_parts)
    
    def _calculate_confidence(self, answer: str, retrieved_docs: List[Dict[str, Any]], question: str) -> float:
        """Calculate confidence score for the generated answer"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if we have good document matches
        if retrieved_docs:
            avg_score = sum(doc.get("score", 0) for doc in retrieved_docs) / len(retrieved_docs)
            confidence += min(avg_score * 0.3, 0.3)  # Max 0.3 boost from document relevance
        
        # Boost confidence for longer, detailed answers
        if len(answer) > 100:
            confidence += 0.1
        
        # Reduce confidence for uncertain language
        uncertain_phrases = ["i'm not sure", "i don't know", "unclear", "cannot determine", "insufficient"]
        if any(phrase in answer.lower() for phrase in uncertain_phrases):
            confidence -= 0.2
        
        # Boost confidence if answer cites sources
        if "[Source:" in answer:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    
    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sources from retrieved documents"""
        sources = set()
        for doc in retrieved_docs:
            if doc.get("source") and doc["source"] != "Unknown":
                sources.add(doc["source"])
        return list(sources)
    
    def _format_context_for_storage(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format retrieved documents for storage in conversation memory"""
        return [
            {
                "content": doc["content"][:500] + ("..." if len(doc["content"]) > 500 else ""),
                "source": doc.get("source", "Unknown"),
                "score": doc.get("score", 0.0),
                "type": doc.get("type", "document")
            }
            for doc in retrieved_docs[:3]  # Store only top 3 for memory efficiency
        ]
    
    async def get_conversation_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the conversation"""
        return conversation_memory.get_session_summary(session_id)
    
    async def search_history(self, query: str, max_results: int = 10) -> List[ConversationTurn]:
        """Search conversation history"""
        return conversation_memory.search_conversation_history(query, max_results)
    
    async def answer_question_with_learning(self, question: str, session_id: Optional[str] = None,
                                          user_id: Optional[str] = None,
                                          enable_adaptation: bool = True) -> QAResponse:
        """Generate an answer with learning and adaptation capabilities"""
        try:
            # Generate the base response
            base_response = await self.answer_question(question, session_id, user_id)
            
            if not enable_adaptation:
                return base_response
            
            # Apply learned adaptations
            context = {
                'sources': [{'source': src} for src in base_response.sources],
                'confidence': base_response.confidence_score,
                'session_id': base_response.session_id
            }
            
            adapted_response = await response_adapter.adapt_response(
                question, base_response, context
            )
            
            # Create enhanced response
            enhanced_response = QAResponse(
                answer=adapted_response.adapted_answer,
                confidence_score=adapted_response.adapted_confidence,
                sources=base_response.sources,
                context_used=base_response.context_used,
                processing_time=base_response.processing_time,
                session_id=base_response.session_id,
                turn_id=base_response.turn_id,
                metadata={
                    **base_response.metadata,
                    "adaptations_applied": adapted_response.adaptations_applied,
                    "adaptation_reasoning": adapted_response.adaptation_reasoning,
                    "improvement_score": adapted_response.improvement_score,
                    "learning_enabled": True
                }
            )
            
            logger.info(f"Applied {len(adapted_response.adaptations_applied)} adaptations to response")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error in learning-enhanced answer generation: {e}")
            return await self.answer_question(question, session_id, user_id)
    
    async def collect_user_feedback(self, session_id: str, turn_id: str, 
                                  feedback_type: str, feedback_value: Any,
                                  feedback_text: Optional[str] = None,
                                  suggested_improvement: Optional[str] = None,
                                  user_correction: Optional[str] = None) -> str:
        """Collect user feedback on a response"""
        try:
            from ..learning_system import FeedbackType
            
            # Get the original question and answer
            session_context = conversation_memory.get_session_context(session_id, max_turns=10)
            target_turn = None
            
            for turn in session_context:
                if turn.turn_id == turn_id:
                    target_turn = turn
                    break
            
            if not target_turn:
                logger.error(f"Turn {turn_id} not found in session {session_id}")
                return ""
            
            # Convert feedback type string to enum
            feedback_type_enum = FeedbackType(feedback_type.lower())
            
            # Collect feedback
            feedback_id = feedback_collector.collect_feedback(
                session_id=session_id,
                turn_id=turn_id,
                question=target_turn.user_question,
                answer=target_turn.system_answer,
                feedback_type=feedback_type_enum,
                feedback_value=feedback_value,
                feedback_text=feedback_text,
                suggested_improvement=suggested_improvement,
                user_correction=user_correction
            )
            
            logger.info(f"Collected feedback {feedback_id} for turn {turn_id}")
            return feedback_id            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return ""
    
    async def trigger_learning_analysis(self) -> Dict[str, Any]:
        """Trigger analysis of feedback patterns and generate new learning rules"""
        try:
            # Analyze feedback patterns with low threshold for testing
            patterns = await learning_engine.analyze_feedback_patterns(min_pattern_support=2)
            
            # Generate adaptation rules
            rules = await learning_engine.generate_adaptation_rules()
            
            return {
                "patterns_discovered": len(patterns),
                "rules_generated": len(rules),
                "learning_statistics": learning_engine.get_learning_statistics(),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in learning analysis: {e}")
            return {"error": str(e)}
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning system performance"""
        try:
            learning_stats = learning_engine.get_learning_statistics()
            adaptation_stats = response_adapter.get_adaptation_statistics()
            
            return {
                "learning_system": learning_stats,
                "adaptation_system": adaptation_stats,
                "integration_status": "active",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {"error": str(e), "integration_status": "error"}
        
        return {
            "patterns_discovered": len(patterns),
            "rules_generated": len(rules),
            "learning_statistics": learning_engine.get_learning_statistics(),
            "analysis_timestamp": datetime.now().isoformat()
        }
    

# Global Q&A engine instance
qa_engine = ContextAwareQAEngine()
