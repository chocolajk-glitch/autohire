"""Ground truth 评测脚本.

跑 11 份简历 x 3 个 JD, 计算系统评分 vs 人工 ground truth 的相关性/准确度.

输出:
- data/eval_results.json: 每个 (jd, resume) 的系统分 vs 真实分
- 整体统计: Pearson / Spearman / MAE / RMSE
- 每个 JD 的 Top-3 命中率

用法:
    python -m eval.run_eval
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import time
from pathlib import Path

from agents.batch import run_batch

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JDS_DIR = DATA_DIR / "jds"
GT_PATH = DATA_DIR / "ground_truth.json"
RESULTS_PATH = DATA_DIR / "eval_results.json"


def _pearson(xs: list[float], ys: list[float]) -> float:
    """皮尔逊相关系数."""
    if len(xs) < 2:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """斯皮尔曼秩相关."""
    if len(xs) < 2:
        return 0.0

    def rank(vs: list[float]) -> list[float]:
        indexed = sorted(enumerate(vs), key=lambda p: p[1])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(indexed):
            j = i
            while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    return _pearson(rank(xs), rank(ys))


def _mae(xs: list[float], ys: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs) if xs else 0.0


def _rmse(xs: list[float], ys: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs)) if xs else 0.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 加载 ground truth
    gt: dict[str, dict[str, int]] = json.loads(GT_PATH.read_text(encoding="utf-8"))
    resume_files = sorted(gt.keys())
    jd_names = ["backend_python_jd", "frontend_vue_jd", "algo_recommendation_jd"]

    logger.info("Loaded %d resumes x %d JDs ground truth", len(resume_files), len(jd_names))

    all_results: list[dict] = []
    t0 = time.time()

    for jd_name in jd_names:
        jd_path = JDS_DIR / f"{jd_name}.txt"
        resume_paths = [RESUMES_DIR / rf for rf in resume_files]
        logger.info("=" * 60)
        logger.info("Running batch for %s with %d resumes", jd_name, len(resume_paths))
        batch_report, summary = run_batch(
            jd_path=jd_path,
            resume_paths=resume_paths,
            enable_reflection=False,
            run_interview_questions=False,
            llm_provider="deepseek",
        )
        # 把结果配上 ground truth
        cand_by_name = {c.candidate_name: c for c in batch_report.candidates}
        for rf in resume_files:
            # 找这份简历对应的人工分
            truth = gt[rf].get(jd_name)
            # 找系统评分
            # 简历文件名 -> candidate_name 映射 (用 "stem" 的前半段, 去 _real_java 后缀)
            stem = Path(rf).stem
            # 候选名: 假设 starg 包含 "zhang_san" 等, ground truth 不知道 candidate_name
            # 改: 简历文件名是 01_zhang_san_strong_backend 这种, 真实姓名在 cand_by_name 里
            # 简单办法: 在所有 candidates 里找, 名字匹配文件名里的关键词
            # 实际: 我们 ground truth 是按"filename"存的, 但 CandidateReport 用 candidate_name
            # 解决: 跑完后, 在 batch_report 里用 candidate_name -> score 映射, 人工 GT 里的简历 "01" 对应 "Wang Lei" 等
            # 因为生成简历时第一个王磊 (wang lei -> filename zhang_san), 所以名称不一致
            # 改: 在 eval 里, 用 fuzzy match
            sys_score = None
            sys_candidate_name = None
            for c in batch_report.candidates:
                # 简单匹配: 文件名前 2 个数字后的第一个单词, 比如 "01_zhang_san_strong_backend" -> "zhang"
                # 但我们 mock 数据里 candidate_name 是 "Wang Lei"
                # 所以只能按 index 对应: 假设 batch 顺序 = resume_files 顺序
                pass
            # 简化: 用 batch_report.candidates 顺序对应 resume_files 顺序
            idx = resume_files.index(rf)
            if idx < len(batch_report.candidates):
                c = batch_report.candidates[idx]
                sys_candidate_name = c.candidate_name
                sys_score = c.match.overall_score
            all_results.append({
                "jd": jd_name,
                "resume_file": rf,
                "candidate_name": sys_candidate_name,
                "system_score": sys_score,
                "ground_truth_score": truth,
                "diff": (sys_score - truth) if (sys_score is not None and truth is not None) else None,
            })
        logger.info("[%s] done in %.1fs, avg=%.1f, hitl=%d",
                    jd_name, summary.duration_seconds, summary.avg_score, summary.hitl_count)

    # 整体统计
    pairs = [(r["system_score"], r["ground_truth_score"]) for r in all_results
             if r["system_score"] is not None and r["ground_truth_score"] is not None]
    sys_scores = [p[0] for p in pairs]
    gt_scores = [p[1] for p in pairs]

    pearson = _pearson(sys_scores, gt_scores)
    spearman = _spearman(sys_scores, gt_scores)
    mae = _mae(sys_scores, gt_scores)
    rmse = _rmse(sys_scores, gt_scores)

    # 每个 JD 的 Top-3 命中率
    top3_hits = {}
    for jd in jd_names:
        jd_pairs = [(r["system_score"], r["ground_truth_score"], r["resume_file"])
                    for r in all_results
                    if r["jd"] == jd and r["system_score"] is not None and r["ground_truth_score"] is not None]
        if len(jd_pairs) < 3:
            continue
        sys_top3 = set(p[2] for p in sorted(jd_pairs, key=lambda x: -x[0])[:3])
        gt_top3 = set(p[2] for p in sorted(jd_pairs, key=lambda x: -x[1])[:3])
        hit = len(sys_top3 & gt_top3)
        top3_hits[jd] = {"hit": hit, "total": 3, "system_top3": sorted(sys_top3), "ground_truth_top3": sorted(gt_top3)}

    summary_report = {
        "total_pairs": len(pairs),
        "total_resumes": len(resume_files),
        "total_jds": len(jd_names),
        "duration_seconds": round(time.time() - t0, 1),
        "overall_metrics": {
            "pearson": round(pearson, 4),
            "spearman": round(spearman, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
        },
        "per_jd_top3_hits": top3_hits,
        "results": all_results,
    }

    RESULTS_PATH.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("=" * 60)
    logger.info("EVAL DONE in %.1fs", summary_report["duration_seconds"])
    logger.info("Pearson=%.4f  Spearman=%.4f  MAE=%.2f  RMSE=%.2f", pearson, spearman, mae, rmse)
    logger.info("Top-3 hits: %s", top3_hits)
    logger.info("Results saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
