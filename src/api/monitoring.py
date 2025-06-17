"""
System monitoring and analytics
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict, deque
import psutil
import time

from loguru import logger

class SystemMonitor:
    """Monitors system performance, usage, and analytics"""
    
    def __init__(self):
        self.metrics_file = Path("data/metrics.json")
        self.metrics_file.parent.mkdir(exist_ok=True)
        
        # In-memory metrics storage
        self.start_time = datetime.now()
        self.request_times = deque(maxlen=1000)  # Last 1000 request times
        self.error_count = 0
        self.request_count = 0
        
        # Usage statistics
        self.question_stats = defaultdict(int)
        self.document_stats = defaultdict(int)
        self.user_activity = defaultdict(list)
        self.feedback_stats = defaultdict(int)
        
        # Load persisted metrics
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from file"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.question_stats.update(data.get('question_stats', {}))
                    self.document_stats.update(data.get('document_stats', {}))
                    self.feedback_stats.update(data.get('feedback_stats', {}))
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            data = {
                'question_stats': dict(self.question_stats),
                'document_stats': dict(self.document_stats),
                'feedback_stats': dict(self.feedback_stats),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    async def track_request(self, processing_time: float):
        """Track API request metrics"""
        self.request_count += 1
        self.request_times.append(processing_time)
    
    async def track_error(self):
        """Track API errors"""
        self.error_count += 1
    
    async def track_question(
        self, 
        user_id: str, 
        question: str, 
        response_time: float, 
        confidence: float
    ):
        """Track Q&A activity"""
        today = datetime.now().strftime('%Y-%m-%d')
        self.question_stats[today] += 1
        self.user_activity[user_id].append({
            'type': 'question',
            'timestamp': datetime.now().isoformat(),
            'response_time': response_time,
            'confidence': confidence
        })
        self._save_metrics()
    
    async def track_document_upload(
        self, 
        user_id: str, 
        filename: str, 
        file_size: int, 
        chunks_created: int
    ):
        """Track document uploads"""
        today = datetime.now().strftime('%Y-%m-%d')
        self.document_stats[today] += 1
        self.user_activity[user_id].append({
            'type': 'upload',
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'file_size': file_size,
            'chunks_created': chunks_created
        })
        self._save_metrics()
    
    async def track_feedback(self, user_id: str, feedback_type: str, value: Any):
        """Track user feedback"""
        self.feedback_stats[feedback_type] += 1
        self.user_activity[user_id].append({
            'type': 'feedback',
            'timestamp': datetime.now().isoformat(),
            'feedback_type': feedback_type,
            'value': value
        })
        self._save_metrics()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        uptime = datetime.now() - self.start_time
        
        # Get system resource usage
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Calculate metrics
        avg_response_time = (
            sum(self.request_times) / len(self.request_times) 
            if self.request_times else 0
        )
        error_rate = (
            self.error_count / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'uptime': str(uptime),
            'memory_usage': memory.percent,
            'cpu_usage': cpu_percent,
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'response_times': list(self.request_times)[-100:],  # Last 100
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        # Calculate totals
        total_questions = sum(self.question_stats.values())
        total_documents = sum(self.document_stats.values())
        total_users = len(self.user_activity)
        
        # Get recent activity (last 30 days)
        recent_questions = []
        today = datetime.now()
        for i in range(30):
            date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            recent_questions.append({
                'date': date,
                'count': self.question_stats.get(date, 0)
            })
        
        # Calculate average metrics
        all_response_times = []
        all_confidences = []
        for activities in self.user_activity.values():
            for activity in activities:
                if activity['type'] == 'question':
                    all_response_times.append(activity['response_time'])
                    all_confidences.append(activity['confidence'])
        
        avg_response_time = (
            sum(all_response_times) / len(all_response_times) 
            if all_response_times else 0
        )
        avg_confidence = (
            sum(all_confidences) / len(all_confidences) 
            if all_confidences else 0
        )
        
        return {
            'total_questions': total_questions,
            'total_documents': total_documents,
            'total_users': total_users,
            'avg_response_time': avg_response_time,
            'avg_confidence_score': avg_confidence,
            'questions_per_day': recent_questions,
            'document_uploads_per_day': [
                {
                    'date': (today - timedelta(days=i)).strftime('%Y-%m-%d'),
                    'count': self.document_stats.get(
                        (today - timedelta(days=i)).strftime('%Y-%m-%d'), 0
                    )
                }
                for i in range(30)
            ]
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        total_feedback = sum(self.feedback_stats.values())
        
        # This would be expanded with actual learning metrics
        return {
            'total_feedback': total_feedback,
            'feedback_distribution': dict(self.feedback_stats),
            'patterns_discovered': 0,  # Would get from learning engine
            'rules_generated': 0,  # Would get from learning engine
            'learning_effectiveness': 0.0  # Would calculate from actual data
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        return {
            'system_stats': self.get_system_stats(),
            'usage_stats': self.get_usage_stats(),
            'learning_stats': self.get_learning_stats(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for frontend"""
        metrics = await self.get_metrics()
        
        # Add additional dashboard-specific data
        dashboard_data = {
            **metrics,
            'alerts': self._get_system_alerts(),
            'recent_activity': self._get_recent_activity(),
            'top_questions': self._get_top_questions(),
            'performance_trends': self._get_performance_trends()
        }
        
        return dashboard_data
    
    def _get_system_alerts(self) -> List[Dict[str, Any]]:
        """Get system alerts"""
        alerts = []
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            alerts.append({
                'type': 'warning',
                'message': f'High memory usage: {memory.percent:.1f}%',
                'timestamp': datetime.now().isoformat()
            })
        
        # Check error rate
        error_rate = (
            self.error_count / self.request_count 
            if self.request_count > 0 else 0
        )
        if error_rate > 0.05:  # 5% error rate
            alerts.append({
                'type': 'error',
                'message': f'High error rate: {error_rate:.1%}',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent user activity"""
        all_activities = []
        for user_id, activities in self.user_activity.items():
            for activity in activities[-10:]:  # Last 10 activities per user
                activity['user_id'] = user_id
                all_activities.append(activity)
        
        # Sort by timestamp and return last 50
        all_activities.sort(key=lambda x: x['timestamp'], reverse=True)
        return all_activities[:50]
    
    def _get_top_questions(self) -> List[str]:
        """Get most frequently asked questions"""
        # This would be implemented with actual question tracking
        return [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are neural networks?",
            "Explain deep learning",
            "What is natural language processing?"
        ]
    
    def _get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends over time"""
        # This would be implemented with time-series data
        return {
            'response_times': list(self.request_times)[-100:],
            'cpu_usage': [psutil.cpu_percent() for _ in range(10)],
            'memory_usage': [psutil.virtual_memory().percent for _ in range(10)]
        }
