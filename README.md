# AutoHire

> 多 Agent 协同的智能招聘筛选系统 · 项目二（实习简历用）

## 项目简介
用户上传 JD + 一批简历 → 多 Agent 自动解析→打分→生成面试题→输出结构化筛选报告，关键节点支持 HR 人工干预。

## 核心架构
- **协调框架**：AutoGen 0.2.x（GroupChat 调度）
- **子协作**：CrewAI（面试出题多角色协作）
- **LLM 工厂**：Qwen / MiniMax / DeepSeek 三家可切换
- **后端**：Python + FastAPI + SSE
- **前端**：Vue 3（最小化）
- **记忆**：SQLite（短期）+ Chroma（长期）

## Agent 分工
| Agent | 框架 | 职责 |
|---|---|---|
| Planner | AutoGen | 任务规划 + 调度 + HITL |
| JD 解析 | AutoGen | JD → 结构化字段 |
| 简历解析 | AutoGen | 简历 → 结构化字段 |
| 匹配度 | AutoGen | 逐维对比 + 反思重判 |
| 面试出题 Crew | **CrewAI** | Researcher + Designer + Reviewer |
| 报告 | AutoGen | 汇总 + 排行榜 |
| HR 协同 | AutoGen | 关键节点 HITL |

## 开发进度
- [x] W1 D1-2：项目结构 + LLM 工厂 + Schemas（进行中）
- [ ] W1 D3-5：JD/简历解析 Agent
- [ ] W1 D6-7：匹配度 Agent + 反思
- [ ] W2 D1-3：Planner + 面试出题 Crew + 报告
- [ ] W2 D4-5：批量 + 排行榜 + HITL
- [ ] W2 D6-7：FastAPI + SSE + Vue
- [ ] W3：评测 + 文档 + 演示视频

## 快速开始
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # 填入 API Key
```
