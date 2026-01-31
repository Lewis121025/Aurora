"""
Aurora Privacy Module
======================

PII (Personally Identifiable Information) detection and redaction.

Usage:
    from aurora.privacy import redact
    
    result = redact("Contact me at john@example.com")
    print(result.redacted_text)  # "Contact me at [REDACTED_EMAIL]"
"""

from aurora.privacy.pii import redact, RedactionResult

__all__ = [
    "redact",
    "RedactionResult",
]
