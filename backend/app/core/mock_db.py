"""Mock in-memory database for testing without MongoDB."""
from typing import Dict, List, Optional
from uuid import uuid4
from datetime import datetime, timezone

# In-memory storage
_users_db: Dict[str, Dict] = {}
_call_logs_db: Dict[str, Dict] = {}


class MockUser:
    """Mock User model for testing."""
    
    def __init__(self, user_id: str = None, email: str = "", username: str = "", hashed_password: str = "", 
                 full_name: Optional[str] = None, family_contact_email: Optional[str] = None, **kwargs):
        self.id = user_id or str(uuid4())
        self.email = email
        self.username = username
        self.hashed_password = hashed_password
        self.full_name = full_name
        self.family_contact_email = family_contact_email
        self.telegram_chat_id = None
        self.is_active = True
        self.created_at = datetime.utcnow()
        self.last_login = None

    async def insert(self):
        """Save user to mock database."""
        _users_db[self.id] = {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'hashed_password': self.hashed_password,
            'full_name': self.full_name,
            'family_contact_email': self.family_contact_email,
            'created_at': self.created_at,
            'last_login': self.last_login,
        }

    async def save(self):
        """Update user in mock database."""
        if self.id in _users_db:
            _users_db[self.id].update({
                'last_login': self.last_login,
            })

    @staticmethod
    async def find_one(query: Dict) -> Optional['MockUser']:
        """Find user by query."""
        for user_data in _users_db.values():
            match = True
            for key, value in query.items():
                if user_data.get(key) != value:
                    match = False
                    break
            if match:
                user = MockUser(
                    user_id=user_data['id'],
                    email=user_data['email'],
                    username=user_data['username'],
                    hashed_password=user_data['hashed_password'],
                    full_name=user_data.get('full_name'),
                    family_contact_email=user_data.get('family_contact_email'),
                )
                user.created_at = user_data['created_at']
                user.last_login = user_data.get('last_login')
                return user
        return None

    @staticmethod
    async def get(user_id: str) -> Optional['MockUser']:
        """Get user by ID."""
        if user_id in _users_db:
            user_data = _users_db[user_id]
            user = MockUser(
                user_id=user_data['id'],
                email=user_data['email'],
                username=user_data['username'],
                hashed_password=user_data['hashed_password'],
                full_name=user_data.get('full_name'),
                family_contact_email=user_data.get('family_contact_email'),
            )
            user.created_at = user_data['created_at']
            user.last_login = user_data.get('last_login')
            return user
        return None


class MockCallLog:
    """Mock CallLog model for testing."""
    
    def __init__(self, call_log_id: str = None, user_id: str = "", **kwargs):
        self.id = call_log_id or str(uuid4())
        self.user_id = user_id
        self.call_start = kwargs.get('call_start', datetime.utcnow())
        self.call_end = kwargs.get('call_end', None)
        self.caller_number = kwargs.get('caller_number', None)
        self.transcript = kwargs.get('transcript', None)
        self.threat_level = kwargs.get('threat_level', "LOW")
        self.deepfake_confidence = kwargs.get('deepfake_confidence', 0.0)
        self.is_deepfake = kwargs.get('is_deepfake', False)
        self.urgency_score = kwargs.get('urgency_score', 0.0)
        self.urgency_detected = kwargs.get('urgency_detected', False)
        self.overall_threat_score = kwargs.get('overall_threat_score', 0.0)
        self.urgency_phrases_found = kwargs.get('urgency_phrases_found', [])
        self.negotiator_strategy = kwargs.get('negotiator_strategy', None)
        self.audio_chunks_received = kwargs.get('audio_chunks_received', 0)
        self.alert_sent = kwargs.get('alert_sent', False)
        self.ai_voice_percentage = kwargs.get('ai_voice_percentage', 0.0)
        self.voice_type = kwargs.get('voice_type', "NATURAL_HUMAN")
        self.pre_recorded_probability = kwargs.get('pre_recorded_probability', 0.0)
        self.voice_verdict = kwargs.get('voice_verdict', None)
        self.acoustic_markers = kwargs.get('acoustic_markers', [])

    async def insert(self):
        """Save call log to mock database."""
        _call_logs_db[self.id] = {
            'id': self.id,
            'user_id': self.user_id,
            'call_start': self.call_start,
            'call_end': self.call_end,
            'caller_number': self.caller_number,
            'transcript': self.transcript,
            'threat_level': self.threat_level,
            'deepfake_confidence': self.deepfake_confidence,
            'is_deepfake': self.is_deepfake,
            'urgency_score': self.urgency_score,
            'urgency_detected': self.urgency_detected,
            'overall_threat_score': self.overall_threat_score,
            'urgency_phrases_found': self.urgency_phrases_found,
            'negotiator_strategy': self.negotiator_strategy,
            'audio_chunks_received': self.audio_chunks_received,
            'alert_sent': self.alert_sent,
            'ai_voice_percentage': self.ai_voice_percentage,
            'voice_type': self.voice_type,
            'pre_recorded_probability': self.pre_recorded_probability,
            'voice_verdict': self.voice_verdict,
            'acoustic_markers': self.acoustic_markers,
        }

    async def save(self):
        """Update or insert call log in mock database."""
        if self.id in _call_logs_db:
            _call_logs_db[self.id].update({
                'threat_level': self.threat_level,
                'deepfake_confidence': self.deepfake_confidence,
                'is_deepfake': self.is_deepfake,
                'urgency_score': self.urgency_score,
                'urgency_detected': self.urgency_detected,
                'overall_threat_score': self.overall_threat_score,
                'urgency_phrases_found': self.urgency_phrases_found,
                'negotiator_strategy': self.negotiator_strategy,
                'audio_chunks_received': self.audio_chunks_received,
                'transcript': self.transcript,
                'alert_sent': self.alert_sent,
                'ai_voice_percentage': self.ai_voice_percentage,
                'voice_type': self.voice_type,
                'pre_recorded_probability': self.pre_recorded_probability,
                'voice_verdict': self.voice_verdict,
                'acoustic_markers': self.acoustic_markers,
            })
        else:
            await self.insert()

    @staticmethod
    def find(query: Dict):
        """Find call logs by query (returns a query builder, not a coroutine).
        NOTE: Must be synchronous to support Beanie-style chaining:
            CallLog.find({...}).sort(...).limit(n).to_list()
        """
        return MockCallLogQuery(query)

    @staticmethod
    async def get(call_log_id: str) -> Optional['MockCallLog']:
        """Get call log by ID."""
        if call_log_id in _call_logs_db:
            log_data = _call_logs_db[call_log_id]
            log = MockCallLog(call_log_id, log_data['user_id'])
            for key, value in log_data.items():
                if key != 'id' and key != 'user_id':
                    setattr(log, key, value)
            return log
        return None


class MockCallLogQuery:
    """Mock query builder for call logs."""
    
    def __init__(self, query: Dict):
        self.query = query
        self.sort_by = None
        self.limit_count = None
        self.results = []
        self._filter()

    def _filter(self):
        """Filter logs by query."""
        self.results = []
        for log_data in _call_logs_db.values():
            match = True
            for key, value in self.query.items():
                if log_data.get(key) != value:
                    match = False
                    break
            if match:
                log = MockCallLog(log_data['id'], log_data['user_id'])
                for k, v in log_data.items():
                    if k not in ['id', 'user_id']:
                        setattr(log, k, v)
                self.results.append(log)

    def sort(self, sort_spec: List):
        """Sort results."""
        if sort_spec:
            field, order = sort_spec[0]
            def _sort_key(item):
                val = getattr(item, field, None)
                if val is None:
                    return ""
                if hasattr(val, "isoformat"):
                    return val.isoformat()
                return str(val)
            self.results.sort(key=_sort_key, reverse=(order == -1))
        return self

    def limit(self, count: int) -> 'MockCallLogQuery':
        """Limit results."""
        self.limit_count = count
        return self

    async def count(self) -> int:
        """Count matched logs."""
        return len(self.results)

    async def to_list(self) -> List[MockCallLog]:
        """Convert to list."""
        if self.limit_count is not None:
            return self.results[:self.limit_count]
        return self.results


def clear_mock_db():
    """Clear all mock data."""
    global _users_db, _call_logs_db
    _users_db.clear()
    _call_logs_db.clear()
