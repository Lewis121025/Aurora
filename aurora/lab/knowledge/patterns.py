"""
AURORA 知识分类规则
========================

关键词、正则模式以及特质关系规则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Pattern, Set, Tuple

from aurora.lab.knowledge.models import KnowledgeType


STATE_KEYWORDS: Set[str] = {
    "住在", "住址", "地址", "电话", "手机", "邮箱", "工作在", "现在是",
    "目前是", "正在", "现居", "任职于", "就职于", "担任", "在做", "邮编",
    "在谷歌", "在微软", "在阿里", "在腾讯", "在百度", "在公司",
    "live in", "lives in", "living in", "address", "phone", "email",
    "work at", "works at", "working at", "currently", "now is", "located in",
    "employed at", "job is", "position is", "residing", "stay at",
}

STATIC_KEYWORDS: Set[str] = {
    "生日", "出生日期", "出生地", "籍贯", "故乡", "血型", "身份证",
    "国籍", "民族", "性别", "出生于", "生于",
    "birthday", "birth date", "birthdate", "born on", "born in",
    "birthplace", "place of birth", "nationality", "blood type",
    "native place", "hometown", "gender", "sex",
}

TRAIT_KEYWORDS: Set[str] = {
    "是...的人", "性格", "特点是", "擅长", "不擅长", "善于", "不善于",
    "天生", "本性", "个性", "脾气", "气质", "为人", "性情",
    "i am", "personality", "good at", "bad at", "tends to be", "tend to be",
    "naturally", "by nature", "character", "temperament", "disposition",
    "trait", "inherently", "fundamentally",
}

TRAIT_WORDS: Set[str] = {
    "耐心", "高效", "细心", "认真", "勤奋", "创意", "逻辑",
    "理性", "感性", "开朗", "内向", "外向", "热情", "冷静",
    "严谨", "灵活", "坚持", "果断", "温和", "敏感", "乐观",
    "谨慎", "大胆", "独立", "合作", "专注", "好奇",
    "patient", "efficient", "careful", "diligent", "creative", "logical",
    "rational", "emotional", "outgoing", "introverted", "extroverted",
    "enthusiastic", "calm", "rigorous", "flexible", "persistent",
    "decisive", "gentle", "sensitive", "optimistic", "cautious",
    "bold", "independent", "collaborative", "focused", "curious",
}

VALUE_KEYWORDS: Set[str] = {
    "价值观", "相信", "坚信", "信仰", "原则", "底线", "准则",
    "重要的是", "最重要", "核心", "根本", "本质", "追求", "理想",
    "value", "believe", "belief", "principle", "bottom line", "standard",
    "important to me", "core", "fundamental", "essence", "pursue", "ideal",
    "integrity", "honesty", "fairness", "justice", "respect", "trust",
}

PREFERENCE_KEYWORDS: Set[str] = {
    "喜欢", "不喜欢", "偏好", "习惯", "喜爱", "讨厌", "爱好",
    "热爱", "痴迷", "钟爱", "厌恶", "反感", "偏爱", "喜好",
    "最爱", "最喜欢", "最讨厌", "倾向于",
    "like", "likes", "prefer", "prefers", "prefer to", "enjoy", "enjoys",
    "hate", "hates", "love", "loves", "dislike", "dislikes", "fond of",
    "favorite", "favourite", "obsessed", "addicted to", "into",
}

BEHAVIOR_KEYWORDS: Set[str] = {
    "通常", "一般", "总是", "经常", "每天", "每周", "习惯于",
    "倾向", "模式", "规律", "作息", "日常", "例行", "定期",
    "usually", "typically", "always", "often", "every day", "every week",
    "tend to", "pattern", "routine", "habit", "regularly", "daily",
    "weekly", "monthly", "schedule",
}


KEYWORD_RULES: Dict[KnowledgeType, Tuple[Set[str], float]] = {
    KnowledgeType.FACTUAL_STATE: (STATE_KEYWORDS, 0.8),
    KnowledgeType.FACTUAL_STATIC: (STATIC_KEYWORDS, 0.9),
    KnowledgeType.IDENTITY_TRAIT: (TRAIT_KEYWORDS, 0.7),
    KnowledgeType.IDENTITY_VALUE: (VALUE_KEYWORDS, 0.85),
    KnowledgeType.PREFERENCE: (PREFERENCE_KEYWORDS, 0.75),
    KnowledgeType.BEHAVIOR_PATTERN: (BEHAVIOR_KEYWORDS, 0.7),
}


@dataclass(frozen=True)
class PatternRule:
    knowledge_type: KnowledgeType
    score: float
    patterns: Tuple[Pattern[str], ...]


PATTERN_RULES: Tuple[PatternRule, ...] = (
    PatternRule(
        knowledge_type=KnowledgeType.FACTUAL_STATE,
        score=1.2,
        patterns=(
            re.compile(r"(.+?)(?:住在|住址是|地址是|电话是|邮箱是|工作在)(.+)", re.IGNORECASE),
            re.compile(r"(.+?)(?:lives? in|works? at|\'s address is|\'s phone is|\'s email is)(.+)", re.IGNORECASE),
            re.compile(r"(.+?)(?:现在是|目前是|currently is|now is)(.+)", re.IGNORECASE),
        ),
    ),
    PatternRule(
        knowledge_type=KnowledgeType.FACTUAL_STATIC,
        score=1.3,
        patterns=(
            re.compile(r"(.+?)(?:的生日是|出生日期是|出生于|生于)(.+)", re.IGNORECASE),
            re.compile(r"(.+?)(?:\'s birthday is|was born on|born in|birthplace is)(.+)", re.IGNORECASE),
        ),
    ),
    PatternRule(
        knowledge_type=KnowledgeType.IDENTITY_TRAIT,
        score=1.0,
        patterns=(
            re.compile(r"(.+?)(?:是一个|是个)(.+?)(?:的人)", re.IGNORECASE),
            re.compile(r"(.+?)(?:很|非常|特别|比较)(\w+)", re.IGNORECASE),
            re.compile(r"(.+?)(?:is a|is an|am a|am an)(.+?)(?:person)", re.IGNORECASE),
            re.compile(r"(.+?)(?:is very|am very|is quite|am quite)(.+)", re.IGNORECASE),
            re.compile(r"(?:我|i)(?:是|am)(.+)", re.IGNORECASE),
        ),
    ),
    PatternRule(
        knowledge_type=KnowledgeType.PREFERENCE,
        score=1.1,
        patterns=(
            re.compile(r"(.+?)(?:喜欢|偏好|热爱|讨厌|不喜欢)(.+)", re.IGNORECASE),
            re.compile(r"(.+?)(?:likes?|prefers?|loves?|hates?|dislikes?)(.+)", re.IGNORECASE),
        ),
    ),
    PatternRule(
        knowledge_type=KnowledgeType.BEHAVIOR_PATTERN,
        score=1.0,
        patterns=(
            re.compile(r"(.+?)(?:通常|一般|总是|经常|习惯)(.+)", re.IGNORECASE),
            re.compile(r"(.+?)(?:usually|typically|always|often|tends? to)(.+)", re.IGNORECASE),
        ),
    ),
)


COMPLEMENTARY_TRAIT_PAIRS: List[Tuple[Set[str], Set[str]]] = [
    ({"耐心", "patient", "慢", "slow", "仔细", "careful"}, {"高效", "efficient", "快", "fast", "迅速", "quick"}),
    ({"内向", "introverted", "安静", "quiet", "独处", "solitary"}, {"社交", "social", "外向", "extroverted", "活跃", "active"}),
    ({"理性", "rational", "逻辑", "logical", "分析", "analytical"}, {"感性", "emotional", "直觉", "intuitive", "感受", "feeling"}),
    ({"严谨", "rigorous", "精确", "precise", "严格", "strict"}, {"灵活", "flexible", "变通", "adaptable", "弹性", "elastic"}),
    ({"独立", "independent", "自主", "autonomous", "单独", "solo"}, {"合作", "collaborative", "团队", "team", "协作", "cooperative"}),
    ({"细节", "detail", "细心", "meticulous", "微观", "micro"}, {"全局", "big picture", "宏观", "macro", "整体", "holistic"}),
    ({"谨慎", "cautious", "保守", "conservative", "稳妥", "prudent"}, {"大胆", "bold", "冒险", "adventurous", "激进", "aggressive"}),
    ({"严肃", "serious", "正经", "formal", "庄重", "solemn"}, {"幽默", "humorous", "风趣", "witty", "轻松", "relaxed"}),
]

CONTRADICTORY_PAIRS: List[Tuple[Set[str], Set[str]]] = [
    ({"诚实", "honest", "真诚", "sincere", "坦诚", "truthful"}, {"说谎", "lying", "liar", "欺骗", "deceptive", "虚伪", "dishonest"}),
    ({"帮助", "helpful", "有益", "beneficial", "有用", "useful"}, {"伤害", "harmful", "有害", "detrimental", "破坏", "destructive"}),
    ({"活着", "alive", "生存", "living", "存活", "surviving"}, {"死亡", "dead", "去世", "deceased", "死去", "passed away"}),
    ({"在场", "present", "在", "here", "出席", "attending"}, {"缺席", "absent", "不在", "away", "离开", "left"}),
]
