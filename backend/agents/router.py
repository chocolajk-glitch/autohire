"""动态路由决策 - 分析 JD 和简历特征, 决定走哪个 Agent 路径.

设计: 4 种路由
1. algorithm_specialist: JD 含算法关键词 -> 算法专项匹配
2. frontend_specialist:  JD 含前端关键词 -> 前端专项匹配
3. ocr_fallback:          简历是图片 / PDF 无文本 -> OCR 解析
4. standard:              默认路径

参考项目一用 LangGraph add_conditional_edges (6 种路由),
项目二用纯函数分类器 (更简单但够用).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.schemas import ParsedJD


# 关键词匹配 (case-insensitive, 兼容中英文)
_ALGO_KEYWORDS = [
    "算法", "推荐", "排序", "召回", "DNN", "A/B", "深度学习", "机器学习",
    "algorithm", "recommend", "ranking", "neural network", "machine learning",
    "PyTorch", "TensorFlow", "FAISS", "embedding",
]

_FRONTEND_KEYWORDS = [
    "前端", "Vue", "React", "Angular", "JavaScript", "TypeScript",
    "前端工程师", "frontend", "前端开发",
    "组件库", "component library", "ECharts", "Pinia", "Webpack", "Vite",
]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass
class RoutingDecision:
    route: str  # algorithm_specialist / frontend_specialist / ocr_fallback / standard
    reason: str
    matched_keywords: list[str] = field(default_factory=list)


def detect_route(
    jd: ParsedJD,
    resume_path: str | Path | None = None,
    resume_text: str | None = None,
) -> RoutingDecision:
    """决定一份简历走哪条路由.

    Args:
        jd: 已解析的 JD
        resume_path: 简历文件路径 (用于判断后缀)
        resume_text: 简历文本 (用于判断 PDF 是否可解析)

    Returns:
        RoutingDecision
    """
    jd_text_parts = [jd.job_title or "", jd.summary or ""]
    for r in jd.requirements:
        jd_text_parts.append(r.description or "")
    jd_text = " ".join(jd_text_parts).lower()

    # 1. 算法专项
    algo_matches = [kw for kw in _ALGO_KEYWORDS if kw.lower() in jd_text]
    if algo_matches:
        return RoutingDecision(
            route="algorithm_specialist",
            reason=f"JD 含算法关键词: {algo_matches[:3]}",
            matched_keywords=algo_matches,
        )

    # 2. 前端专项
    fe_matches = [kw for kw in _FRONTEND_KEYWORDS if kw.lower() in jd_text]
    if fe_matches:
        return RoutingDecision(
            route="frontend_specialist",
            reason=f"JD 含前端关键词: {fe_matches[:3]}",
            matched_keywords=fe_matches,
        )

    # 3. OCR fallback (图片文件 或 PDF 无文本)
    if resume_path is not None:
        ext = Path(resume_path).suffix.lower()
        if ext in _IMAGE_EXTENSIONS:
            return RoutingDecision(
                route="ocr_fallback",
                reason=f"简历是图片文件 ({ext}), 需 OCR 解析",
            )
    if resume_path is not None and resume_text is not None:
        ext = Path(resume_path).suffix.lower()
        if ext == ".pdf" and len(resume_text.strip()) < 100:
            return RoutingDecision(
                route="ocr_fallback",
                reason=f"PDF 文件但文本极少 ({len(resume_text)} 字符), 可能是扫描件",
            )

    # 4. 默认
    return RoutingDecision(
        route="standard",
        reason="JD 无特殊关键词, 走标准匹配路径",
    )