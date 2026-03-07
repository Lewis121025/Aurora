"""
Aurora 隐私模块
======================

PII（个人可识别信息）检测和编辑。

使用方法:
    from aurora.privacy import redact

    result = redact("Contact me at john@example.com")
    print(result.redacted_text)  # "Contact me at [REDACTED_EMAIL]"
"""

from aurora.privacy.pii import redact, RedactionResult

__all__ = [
    "redact",
    "RedactionResult",
]
