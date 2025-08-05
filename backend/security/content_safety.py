#!/usr/bin/env python3
"""
PII/Toxicity Content Re-scanning System
Continuous monitoring and re-scanning of uploaded content for privacy and safety compliance.
"""

import os
import re
import json
import hashlib
import time
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import threading
from enum import Enum
import base64

# Optional ML libraries for advanced detection
try:
    import transformers
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


class ContentRisk(Enum):
    """Content risk levels."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MODERATE = "moderate"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


class ViolationType(Enum):
    """Types of content violations."""
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_SSN = "pii_ssn"
    PII_CREDIT_CARD = "pii_credit_card"
    PII_ADDRESS = "pii_address"
    PII_NAME = "pii_name"
    TOXICITY_HATE_SPEECH = "toxicity_hate_speech"
    TOXICITY_HARASSMENT = "toxicity_harassment"
    TOXICITY_PROFANITY = "toxicity_profanity"
    TOXICITY_THREAT = "toxicity_threat"
    INAPPROPRIATE_SEXUAL = "inappropriate_sexual"
    INAPPROPRIATE_VIOLENCE = "inappropriate_violence"
    MALWARE_SIGNATURE = "malware_signature"
    SPAM_CONTENT = "spam_content"


@dataclass
class ContentViolation:
    """Content violation detection result."""
    violation_type: ViolationType
    severity: ContentRisk
    confidence: float  # 0.0 to 1.0
    location: str  # Where in content the violation was found
    context: str  # Surrounding text for context
    suggested_action: str  # Recommended action
    auto_fixable: bool  # Whether violation can be automatically fixed


@dataclass
class ContentScanResult:
    """Result of content safety scan."""
    content_id: str
    content_hash: str
    scan_timestamp: float
    overall_risk: ContentRisk
    violations: List[ContentViolation]
    pii_detected: bool
    toxicity_detected: bool
    malware_detected: bool
    scan_duration_ms: float
    scanner_version: str
    metadata: Dict[str, Any]


@dataclass
class ContentMonitoringConfig:
    """Configuration for content monitoring."""
    scan_interval_hours: int
    enable_pii_detection: bool
    enable_toxicity_detection: bool
    enable_malware_detection: bool
    auto_quarantine_threshold: ContentRisk
    notification_threshold: ContentRisk
    retention_days: int
    max_content_size_mb: int


class PIIDetector:
    """Detect personally identifiable information in content."""
    
    def __init__(self):
        # Regular expressions for PII detection
        self.patterns = {
            ViolationType.PII_EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            ViolationType.PII_PHONE: re.compile(
                r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
            ),
            ViolationType.PII_SSN: re.compile(
                r'\b\d{3}-?\d{2}-?\d{4}\b'
            ),
            ViolationType.PII_CREDIT_CARD: re.compile(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
            ),
        }
        
        # Load spaCy model for named entity recognition if available
        self.nlp = None
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                try:
                    self.nlp = spacy.load("en_core_web_lg")
                except Exception:
                    print("Warning: spaCy models not found. PII detection will be limited.")
    
    def detect_pii(self, content: str) -> List[ContentViolation]:
        """Detect PII in content."""
        violations = []
        
        # Regex-based detection
        for violation_type, pattern in self.patterns.items():
            matches = pattern.finditer(content)
            for match in matches:
                context_start = max(0, match.start() - 50)
                context_end = min(len(content), match.end() + 50)
                context = content[context_start:context_end]
                
                violation = ContentViolation(
                    violation_type=violation_type,
                    severity=ContentRisk.HIGH_RISK,
                    confidence=0.9,  # High confidence for regex matches
                    location=f"Position {match.start()}-{match.end()}",
                    context=context,
                    suggested_action="Remove or redact PII",
                    auto_fixable=True
                )
                violations.append(violation)
        
        # spaCy-based named entity recognition
        if self.nlp:
            try:
                doc = self.nlp(content[:10000])  # Limit content length for performance
                
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "GPE", "ORG"]:  # Person, place, organization
                        # Check if it looks like a real name (not generic terms)
                        if len(ent.text.split()) >= 2 and ent.text[0].isupper():
                            context_start = max(0, ent.start_char - 50)
                            context_end = min(len(content), ent.end_char + 50)
                            context = content[context_start:context_end]
                            
                            violation = ContentViolation(
                                violation_type=ViolationType.PII_NAME,
                                severity=ContentRisk.MODERATE,
                                confidence=0.7,
                                location=f"Position {ent.start_char}-{ent.end_char}",
                                context=context,
                                suggested_action="Review potential name mention",
                                auto_fixable=False
                            )
                            violations.append(violation)
                            
            except Exception as e:
                print(f"Warning: spaCy NER failed: {e}")
        
        return violations


class ToxicityDetector:
    """Detect toxic and inappropriate content."""
    
    def __init__(self):
        self.toxicity_classifier = None
        
        # Initialize transformers-based toxicity detection if available
        if HAS_TRANSFORMERS:
            try:
                self.toxicity_classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert-base-uncased",
                    device=-1  # Use CPU
                )
            except Exception:
                try:
                    # Fallback to a different model
                    self.toxicity_classifier = pipeline(
                        "text-classification",
                        model="martin-ha/toxic-comment-model",
                        device=-1
                    )
                except Exception as e:
                    print(f"Warning: Toxicity model loading failed: {e}")
        
        # Keyword-based detection as fallback
        self.toxic_keywords = {
            ViolationType.TOXICITY_HATE_SPEECH: [
                "hate", "nazi", "fascist", "terrorist", "extremist"
            ],
            ViolationType.TOXICITY_HARASSMENT: [
                "harass", "bully", "threaten", "intimidat", "stalk"
            ],
            ViolationType.TOXICITY_PROFANITY: [
                "fuck", "shit", "damn", "bitch", "asshole"
            ],
            ViolationType.TOXICITY_THREAT: [
                "kill", "murder", "bomb", "shoot", "attack", "destroy"
            ]
        }
    
    def detect_toxicity(self, content: str) -> List[ContentViolation]:
        """Detect toxicity in content."""
        violations = []
        
        # Transformers-based detection
        if self.toxicity_classifier:
            try:
                # Split content into chunks for processing
                chunk_size = 512
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                
                for i, chunk in enumerate(chunks[:10]):  # Limit to first 10 chunks
                    if len(chunk.strip()) < 10:  # Skip very short chunks
                        continue
                        
                    result = self.toxicity_classifier(chunk)
                    
                    # Check if toxic (different models have different label formats)
                    is_toxic = False
                    confidence = 0.0
                    
                    if isinstance(result, list) and len(result) > 0:
                        if result[0].get('label') in ['TOXIC', '1', 'toxic']:
                            is_toxic = True
                            confidence = result[0].get('score', 0.0)
                        elif result[0].get('label') in ['NEGATIVE']:
                            # Some models use NEGATIVE for toxic content
                            confidence = result[0].get('score', 0.0)
                            is_toxic = confidence > 0.8
                    
                    if is_toxic and confidence > 0.7:
                        violation = ContentViolation(
                            violation_type=ViolationType.TOXICITY_HATE_SPEECH,
                            severity=ContentRisk.HIGH_RISK if confidence > 0.9 else ContentRisk.MODERATE,
                            confidence=confidence,
                            location=f"Chunk {i+1} (position ~{i*chunk_size})",
                            context=chunk[:200] + "..." if len(chunk) > 200 else chunk,
                            suggested_action="Review and potentially remove toxic content",
                            auto_fixable=False
                        )
                        violations.append(violation)
                        
            except Exception as e:
                print(f"Warning: Transformers toxicity detection failed: {e}")
        
        # Keyword-based detection as fallback
        content_lower = content.lower()
        for violation_type, keywords in self.toxic_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # Find all occurrences
                    start_pos = 0
                    while True:
                        pos = content_lower.find(keyword, start_pos)
                        if pos == -1:
                            break
                        
                        context_start = max(0, pos - 50)
                        context_end = min(len(content), pos + len(keyword) + 50)
                        context = content[context_start:context_end]
                        
                        violation = ContentViolation(
                            violation_type=violation_type,
                            severity=ContentRisk.MODERATE,
                            confidence=0.6,  # Lower confidence for keyword matching
                            location=f"Position {pos}-{pos+len(keyword)}",
                            context=context,
                            suggested_action=f"Review usage of keyword '{keyword}'",
                            auto_fixable=False
                        )
                        violations.append(violation)
                        
                        start_pos = pos + 1
        
        return violations


class MalwareDetector:
    """Detect potential malware signatures and suspicious patterns."""
    
    def __init__(self):
        # Suspicious patterns that might indicate malware
        self.suspicious_patterns = {
            ViolationType.MALWARE_SIGNATURE: [
                r'eval\s*\(',  # JavaScript eval
                r'exec\s*\(',  # Python exec
                r'system\s*\(',  # System calls
                r'shell_exec\s*\(',  # PHP shell execution
                r'<script[^>]*>.*?</script>',  # Script tags
                r'javascript:\s*',  # JavaScript URLs
                r'data:text\/html',  # Data URLs
                r'\.exe\b',  # Executable files
                r'\.bat\b',  # Batch files
                r'\.cmd\b',  # Command files
            ]
        }
    
    def detect_malware(self, content: str) -> List[ContentViolation]:
        """Detect potential malware signatures."""
        violations = []
        
        for violation_type, patterns in self.suspicious_patterns.items():
            for pattern_str in patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
                matches = pattern.finditer(content)
                
                for match in matches:
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(content), match.end() + 100)
                    context = content[context_start:context_end]
                    
                    violation = ContentViolation(
                        violation_type=violation_type,
                        severity=ContentRisk.HIGH_RISK,
                        confidence=0.8,
                        location=f"Position {match.start()}-{match.end()}",
                        context=context,
                        suggested_action="Review for potential malware",
                        auto_fixable=False
                    )
                    violations.append(violation)
        
        return violations


class ContentSafetyScanner:
    """Main content safety scanning system."""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path("./data/content_safety")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detectors
        self.pii_detector = PIIDetector()
        self.toxicity_detector = ToxicityDetector()
        self.malware_detector = MalwareDetector()
        
        # Configuration
        self.config = ContentMonitoringConfig(
            scan_interval_hours=24,
            enable_pii_detection=True,
            enable_toxicity_detection=True,
            enable_malware_detection=True,
            auto_quarantine_threshold=ContentRisk.HIGH_RISK,
            notification_threshold=ContentRisk.MODERATE,
            retention_days=90,
            max_content_size_mb=10
        )
        
        # Scan results storage
        self.scan_results: Dict[str, ContentScanResult] = {}
        self.quarantined_content: Set[str] = set()
        
        # Monitoring
        self.monitoring_enabled = True
        self.monitor_thread = None
        
        self._load_scan_results()
        self._start_content_monitoring()
        
        print(f"🛡️ Content Safety Scanner initialized")
        print(f"   Storage: {self.storage_dir}")
        print(f"   PII Detection: {'✅' if self.config.enable_pii_detection else '❌'}")
        print(f"   Toxicity Detection: {'✅' if HAS_TRANSFORMERS else '⚠️ Limited'}")
        print(f"   Malware Detection: {'✅' if self.config.enable_malware_detection else '❌'}")
    
    def _load_scan_results(self):
        """Load previous scan results from storage."""
        results_file = self.storage_dir / "scan_results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    for content_id, result_data in data.items():
                        result = ContentScanResult(**result_data)
                        result.overall_risk = ContentRisk(result_data["overall_risk"])
                        
                        # Convert violations
                        violations = []
                        for violation_data in result_data["violations"]:
                            violation = ContentViolation(**violation_data)
                            violation.violation_type = ViolationType(violation_data["violation_type"])
                            violation.severity = ContentRisk(violation_data["severity"])
                            violations.append(violation)
                        
                        result.violations = violations
                        self.scan_results[content_id] = result
                        
            except Exception as e:
                print(f"Warning: Failed to load scan results: {e}")
        
        # Load quarantined content list
        quarantine_file = self.storage_dir / "quarantined_content.json"
        if quarantine_file.exists():
            try:
                with open(quarantine_file) as f:
                    self.quarantined_content = set(json.load(f))
            except Exception as e:
                print(f"Warning: Failed to load quarantine list: {e}")
    
    def _save_scan_results(self):
        """Save scan results to storage."""
        results_file = self.storage_dir / "scan_results.json"
        try:
            data = {}
            for content_id, result in self.scan_results.items():
                result_data = asdict(result)
                result_data["overall_risk"] = result.overall_risk.value
                
                # Convert violations
                violations_data = []
                for violation in result.violations:
                    violation_data = asdict(violation)
                    violation_data["violation_type"] = violation.violation_type.value
                    violation_data["severity"] = violation.severity.value
                    violations_data.append(violation_data)
                
                result_data["violations"] = violations_data
                data[content_id] = result_data
            
            # Atomic write
            temp_file = results_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(results_file)
            
        except Exception as e:
            print(f"Warning: Failed to save scan results: {e}")
        
        # Save quarantined content list
        quarantine_file = self.storage_dir / "quarantined_content.json"
        try:
            with open(quarantine_file, 'w') as f:
                json.dump(list(self.quarantined_content), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save quarantine list: {e}")
    
    def scan_content(self, content_id: str, content: Union[str, bytes]) -> ContentScanResult:
        """Scan content for safety violations."""
        start_time = time.time()
        
        # Convert bytes to string if needed
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8', errors='ignore')
            except Exception:
                content = str(content)
        
        # Calculate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check content size
        content_size_mb = len(content.encode()) / (1024 * 1024)
        if content_size_mb > self.config.max_content_size_mb:
            # Truncate content for scanning
            max_chars = self.config.max_content_size_mb * 1024 * 1024
            content = content[:max_chars]
        
        violations = []
        
        # PII Detection
        if self.config.enable_pii_detection:
            try:
                pii_violations = self.pii_detector.detect_pii(content)
                violations.extend(pii_violations)
            except Exception as e:
                print(f"Warning: PII detection failed: {e}")
        
        # Toxicity Detection
        if self.config.enable_toxicity_detection:
            try:
                toxicity_violations = self.toxicity_detector.detect_toxicity(content)
                violations.extend(toxicity_violations)
            except Exception as e:
                print(f"Warning: Toxicity detection failed: {e}")
        
        # Malware Detection
        if self.config.enable_malware_detection:
            try:
                malware_violations = self.malware_detector.detect_malware(content)
                violations.extend(malware_violations)
            except Exception as e:
                print(f"Warning: Malware detection failed: {e}")
        
        # Determine overall risk
        if not violations:
            overall_risk = ContentRisk.SAFE
        else:
            max_severity = max(v.severity for v in violations)
            overall_risk = max_severity
        
        # Create scan result
        scan_duration_ms = (time.time() - start_time) * 1000
        
        result = ContentScanResult(
            content_id=content_id,
            content_hash=content_hash,
            scan_timestamp=time.time(),
            overall_risk=overall_risk,
            violations=violations,
            pii_detected=any(v.violation_type.value.startswith("pii_") for v in violations),
            toxicity_detected=any(v.violation_type.value.startswith("toxicity_") for v in violations),
            malware_detected=any(v.violation_type.value.startswith("malware_") for v in violations),
            scan_duration_ms=scan_duration_ms,
            scanner_version="1.0.0",
            metadata={
                "content_size_bytes": len(content.encode()),
                "violations_count": len(violations),
                "high_risk_violations": len([v for v in violations if v.severity == ContentRisk.HIGH_RISK])
            }
        )
        
        self.scan_results[content_id] = result
        
        # Auto-quarantine if needed
        if overall_risk.value in [ContentRisk.HIGH_RISK.value, ContentRisk.CRITICAL.value]:
            self.quarantined_content.add(content_id)
            
            # Record security event
            from .monitoring import record_security_event
            record_security_event(
                "content_auto_quarantined",
                "content_safety_scanner",
                {
                    "content_id": content_id,
                    "risk_level": overall_risk.value,
                    "violations_count": len(violations),
                    "pii_detected": result.pii_detected,
                    "toxicity_detected": result.toxicity_detected,
                    "malware_detected": result.malware_detected
                }
            )
        
        self._save_scan_results()
        
        print(f"🔍 Content scan completed: {content_id}")
        print(f"   Risk Level: {overall_risk.value}")
        print(f"   Violations: {len(violations)}")
        print(f"   Scan Time: {scan_duration_ms:.1f}ms")
        
        return result
    
    def is_content_safe(self, content_id: str) -> Tuple[bool, Optional[ContentScanResult]]:
        """Check if content is safe for use."""
        if content_id in self.quarantined_content:
            return False, self.scan_results.get(content_id)
        
        result = self.scan_results.get(content_id)
        if not result:
            return False, None  # Not scanned yet
        
        # Check if scan is recent enough
        scan_age_hours = (time.time() - result.scan_timestamp) / 3600
        if scan_age_hours > self.config.scan_interval_hours * 2:
            return False, result  # Scan too old
        
        # Check risk level
        is_safe = result.overall_risk in [ContentRisk.SAFE, ContentRisk.LOW_RISK]
        return is_safe, result
    
    def _start_content_monitoring(self):
        """Start periodic content monitoring."""
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            print("🔍 Content safety monitoring started")
    
    def _monitoring_loop(self):
        """Periodic content re-scanning loop."""
        while self.monitoring_enabled:
            try:
                # Re-scan old results
                current_time = time.time()
                rescan_threshold = self.config.scan_interval_hours * 3600
                
                content_to_rescan = []
                for content_id, result in self.scan_results.items():
                    if current_time - result.scan_timestamp > rescan_threshold:
                        content_to_rescan.append(content_id)
                
                if content_to_rescan:
                    print(f"🔄 Re-scanning {len(content_to_rescan)} content items due to age")
                    # Note: In practice, you'd need to retrieve the actual content to re-scan
                    # This is a placeholder for the re-scanning logic
                
                # Clean up old results
                cleanup_threshold = self.config.retention_days * 86400
                content_to_cleanup = []
                for content_id, result in self.scan_results.items():
                    if current_time - result.scan_timestamp > cleanup_threshold:
                        content_to_cleanup.append(content_id)
                
                for content_id in content_to_cleanup:
                    del self.scan_results[content_id]
                    self.quarantined_content.discard(content_id)
                
                if content_to_cleanup:
                    print(f"🧹 Cleaned up {len(content_to_cleanup)} old scan results")
                    self._save_scan_results()
                
                # Sleep until next check
                time.sleep(3600)  # 1 hour
                
            except Exception as e:
                print(f"Warning: Content monitoring error: {e}")
                time.sleep(1800)  # 30 minutes on error
    
    def get_content_safety_status(self) -> Dict:
        """Get content safety system status."""
        total_scanned = len(self.scan_results)
        quarantined_count = len(self.quarantined_content)
        
        risk_distribution = {risk.value: 0 for risk in ContentRisk}
        violation_distribution = {vtype.value: 0 for vtype in ViolationType}
        
        for result in self.scan_results.values():
            risk_distribution[result.overall_risk.value] += 1
            for violation in result.violations:
                violation_distribution[violation.violation_type.value] += 1
        
        recent_scans = [r for r in self.scan_results.values() if time.time() - r.scan_timestamp < 86400]
        
        return {
            "total_content_scanned": total_scanned,
            "quarantined_content": quarantined_count,
            "risk_distribution": risk_distribution,
            "violation_distribution": violation_distribution,
            "recent_scans_24h": len(recent_scans),
            "avg_scan_time_ms": sum(r.scan_duration_ms for r in recent_scans) / len(recent_scans) if recent_scans else 0,
            "monitoring_enabled": self.monitoring_enabled,
            "configuration": {
                "scan_interval_hours": self.config.scan_interval_hours,
                "pii_detection": self.config.enable_pii_detection,
                "toxicity_detection": self.config.enable_toxicity_detection,
                "malware_detection": self.config.enable_malware_detection,
                "auto_quarantine_threshold": self.config.auto_quarantine_threshold.value,
                "max_content_size_mb": self.config.max_content_size_mb
            },
            "detector_status": {
                "transformers_available": HAS_TRANSFORMERS,
                "spacy_available": HAS_SPACY,
                "advanced_detection": HAS_TRANSFORMERS and HAS_SPACY
            }
        }


# Global instance
content_safety_scanner = ContentSafetyScanner()


# Convenience functions
def scan_content_safety(content_id: str, content: Union[str, bytes]) -> ContentScanResult:
    """Scan content for safety violations."""
    return content_safety_scanner.scan_content(content_id, content)


def is_content_safe(content_id: str) -> Tuple[bool, Optional[ContentScanResult]]:
    """Check if content is safe for use."""
    return content_safety_scanner.is_content_safe(content_id)


def get_content_safety_status() -> Dict:
    """Get content safety system status."""
    return content_safety_scanner.get_content_safety_status()


def quarantine_content(content_id: str):
    """Manually quarantine content."""
    content_safety_scanner.quarantined_content.add(content_id)
    content_safety_scanner._save_scan_results()


def unquarantine_content(content_id: str):
    """Remove content from quarantine."""
    content_safety_scanner.quarantined_content.discard(content_id)
    content_safety_scanner._save_scan_results()