"""生成 10 份模拟简历 PDF + 3 个 JD, 用于批量评估和 ground truth 评测.

设计:
- 10 份简历覆盖: 前端 / 后端 / 算法 / 全栈 / 学生, 工作年限 0-8 年
- 3 个 JD: 后端 (Python) / 前端 (Vue) / 算法 (推荐系统)
- 人工标注的 ground truth 分数 (0-100) 用于 W3 评测
"""
import json
import random
from pathlib import Path

import fitz

random.seed(42)

DATA_DIR = Path(__file__).parent
RESUMES_DIR = DATA_DIR / "resumes"
JDS_DIR = DATA_DIR / "jds"
RESUMES_DIR.mkdir(parents=True, exist_ok=True)
JDS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 10 份简历 (每份 = 文本内容)
# ============================================================
RESUMES = [
    # 1. 强后端匹配 (Python 3 年, 完美匹配 JD 1)
    {
        "filename": "01_zhang_san_strong_backend.pdf",
        "content": """Wang Lei
Email: wanglei@example.com  |  Phone: 138-0001-0001  |  Location: Shanghai

EDUCATION
Shanghai Jiao Tong University, Bachelor of Computer Science, 2019 - 2023

WORK EXPERIENCE
ByteDance (2023 - Present)
Backend Engineer
- Designed and built high-throughput ad ranking system serving 200M+ users
- Led migration of Python 2 monolith to Python 3 + FastAPI microservices
- Built internal tools using LangGraph and CrewAI for multi-agent workflows
- Mentored 2 junior engineers

PROJECTS
AutoHire (Multi-Agent Recruitment Platform) | 2025 - 2026
- Tech Stack: Python, FastAPI, LangGraph, AutoGen, CrewAI, PostgreSQL, Redis, Docker, Vue 3
- Role: Lead developer
- Description: Built a multi-agent resume screening system with Planner, 5 AutoGen agents,
  and 1 CrewAI 3-role crew. Implemented self-reflection, HITL, and structured output validation.
- Duration: 6 months

Distributed Crawler Framework | 2024
- Tech Stack: Python, asyncio, Redis, RabbitMQ, PostgreSQL, Docker
- Description: Built distributed crawler handling 50M pages/day
- Duration: 3 months

SKILLS
Python, FastAPI, LangGraph, AutoGen, CrewAI, PostgreSQL, Redis, Docker, Kubernetes,
Vue 3, MySQL, ChromaDB, asyncio, RabbitMQ

SELF SUMMARY
Backend engineer with 3 years of experience, focused on AI infrastructure and
multi-agent systems. Strong in Python ecosystem, LLM application development, and
high-throughput system design.
""",
        "ground_truth": {
            "backend_python_jd": 90,
            "frontend_vue_jd": 45,
            "algo_recommendation_jd": 55,
        },
    },
    # 2. 中等后端 (Java 转 Python, 2 年经验, 缺 LangGraph)
    {
        "filename": "02_li_si_mid_backend.pdf",
        "content": """Li Si
Email: lisi@example.com  |  Phone: 138-0002-0002  |  Location: Beijing

EDUCATION
Beijing University of Posts and Telecommunications, Bachelor of Software Engineering, 2020 - 2024

WORK EXPERIENCE
Meituan (2024 - Present)
Backend Engineer
- Built order management system using Spring Boot + MySQL
- Migrated some services from Java to Python for ML pipelines
- Used Redis for caching and Kafka for async messaging

PROJECTS
Food Delivery Backend Optimization | 2024
- Tech Stack: Java, Spring Boot, MySQL, Redis, Kafka
- Description: Optimized order processing latency from 200ms to 50ms
- Duration: 4 months

SKILLS
Java, Spring Boot, Python (basic), MySQL, Redis, Kafka, Docker, Git

SELF SUMMARY
Backend engineer with 1 year of experience, primarily Java/Spring Boot stack.
Recently transitioning to Python for ML/data work.
""",
        "ground_truth": {
            "backend_python_jd": 35,
            "frontend_vue_jd": 20,
            "algo_recommendation_jd": 25,
        },
    },
    # 3. 强前端 (Vue 3 年, 完美匹配 JD 2)
    {
        "filename": "03_zhao_wu_strong_frontend.pdf",
        "content": """Zhao Wu
Email: zhaowu@example.com  |  Phone: 138-0003-0003  |  Location: Hangzhou

EDUCATION
Zhejiang University, Bachelor of Information Engineering, 2019 - 2023

WORK EXPERIENCE
Alibaba (2023 - Present)
Frontend Engineer
- Built large-scale e-commerce SPA using Vue 3 + TypeScript + Pinia
- Designed and maintained component library used by 50+ internal apps
- Implemented micro-frontend architecture with Module Federation
- Mentored 3 junior engineers on Vue best practices

PROJECTS
Vue 3 Component Library | 2024
- Tech Stack: Vue 3, TypeScript, Vite, Storybook, Jest
- Description: Built 60+ reusable components, 90% test coverage
- Duration: 5 months

Real-time Dashboard | 2025
- Tech Stack: Vue 3, WebSocket, ECharts, Tailwind CSS
- Description: Real-time monitoring dashboard for logistics operations
- Duration: 2 months

SKILLS
Vue 3, TypeScript, JavaScript, HTML, CSS, Pinia, Vite, Webpack, ECharts,
Tailwind CSS, Jest, Cypress, Node.js (basic)

SELF SUMMARY
Frontend engineer with 3 years of experience, Vue 3 specialist. Strong in
component design, state management, and performance optimization.
""",
        "ground_truth": {
            "backend_python_jd": 30,
            "frontend_vue_jd": 92,
            "algo_recommendation_jd": 20,
        },
    },
    # 4. 强算法 (推荐系统, 完美匹配 JD 3)
    {
        "filename": "04_sun_li_strong_algo.pdf",
        "content": """Sun Li
Email: sunli@example.com  |  Phone: 138-0004-0004  |  Location: Beijing

EDUCATION
Tsinghua University, Master of Computer Science (Machine Learning), 2020 - 2023
Tsinghua University, Bachelor of Mathematics, 2016 - 2020

WORK EXPERIENCE
Kuaishou (2023 - Present)
Senior Algorithm Engineer
- Designed and deployed short-video recommendation system using DNN + multi-task learning
- Built feature store serving 1B+ user behavior features with < 50ms latency
- Led A/B testing framework, drove 15% CTR improvement
- Published 2 papers at RecSys and SIGIR

PROJECTS
Deep Interest Network (DIN) for E-commerce | 2022
- Tech Stack: TensorFlow, Python, Spark, Kafka
- Description: Implemented DIN for click-through rate prediction, 8% AUC gain
- Duration: 4 months

Multi-modal Content Understanding | 2024
- Tech Stack: PyTorch, CLIP, FAISS, Ray
- Description: Built video-text-image joint embedding for recommendation
- Duration: 6 months

SKILLS
Python, PyTorch, TensorFlow, Spark, Kafka, FAISS, SQL, RecSys, Multi-task Learning,
A/B Testing, C++ (basic), CUDA (basic)

SELF SUMMARY
Algorithm engineer with 3 years of experience, specialized in recommendation
systems. Strong in DNN, feature engineering, and large-scale system deployment.
""",
        "ground_truth": {
            "backend_python_jd": 50,
            "frontend_vue_jd": 25,
            "algo_recommendation_jd": 95,
        },
    },
    # 5. 全栈 (中等, 啥都会一点)
    {
        "filename": "05_zhou_qi_fullstack.pdf",
        "content": """Zhou Qi
Email: zhouqi@example.com  |  Phone: 138-0005-0005  |  Location: Shenzhen

EDUCATION
Sun Yat-sen University, Bachelor of Software Engineering, 2018 - 2022

WORK EXPERIENCE
Tencent (2022 - Present)
Full-stack Engineer
- Built internal CRM system with Vue 3 frontend and Python FastAPI backend
- Implemented basic recommendation feature using collaborative filtering
- Deployed services on Kubernetes, wrote Helm charts
- Some LLM API integration work (calling OpenAI APIs)

PROJECTS
Internal CRM | 2022 - 2023
- Tech Stack: Vue 3, FastAPI, PostgreSQL, Redis
- Description: Built CRM used by 200+ sales reps
- Duration: 8 months

SKILLS
Python, FastAPI, Vue 3, TypeScript, PostgreSQL, Redis, Docker, Kubernetes,
Git, basic ML

SELF SUMMARY
Full-stack engineer with 3 years of experience. Comfortable across the stack,
recently interested in LLM applications.
""",
        "ground_truth": {
            "backend_python_jd": 60,
            "frontend_vue_jd": 60,
            "algo_recommendation_jd": 40,
        },
    },
    # 6. 应届生 (0 经验, 强算法背景)
    {
        "filename": "06_wu_junior_algo.pdf",
        "content": """Wu Xiao
Email: wuxiao@example.com  |  Phone: 138-0006-0006  |  Location: Shanghai

EDUCATION
Fudan University, Master of Computer Science (NLP), 2023 - 2026 (expected)
Fudan University, Bachelor of Computer Science, 2019 - 2023

PUBLICATIONS
- "Improving BERT for Long Document Classification" (ACL 2024 workshop)
- "Cross-lingual Transfer Learning for Low-resource NLP" (EMNLP 2024)

PROJECTS
Thesis: Multi-modal Document Retrieval | 2025
- Tech Stack: PyTorch, Hugging Face, FAISS, LangChain
- Description: Built RAG system with custom embedding model
- Duration: 8 months

Course Project: RecSys Competition | 2024
- Tech Stack: Python, XGBoost, LightGBM
- Description: Top 5% in Tianchi RecSys competition
- Duration: 2 months

SKILLS
Python, PyTorch, Hugging Face, FAISS, LangChain, SQL, basic Java

SELF SUMMARY
MS student graduating 2026, focused on NLP and information retrieval.
Strong in ML fundamentals, limited industry experience.
""",
        "ground_truth": {
            "backend_python_jd": 40,
            "frontend_vue_jd": 15,
            "algo_recommendation_jd": 65,
        },
    },
    # 7. 老后端 (8 年 Java, 不熟 Python)
    {
        "filename": "07_zheng_ba_senior_java.pdf",
        "content": """Zheng Jiu
Email: zhengjiu@example.com  |  Phone: 138-0007-0007  |  Location: Shanghai

EDUCATION
Huazhong University of Science and Technology, Bachelor of Computer Science, 2014 - 2018

WORK EXPERIENCE
Alibaba (2018 - 2022)
Senior Backend Engineer
- Built high-concurrency order system using Java + Spring Cloud
- Led design of distributed ID generation service (1M+ QPS)
- Wrote 20+ technical blog posts on system design

Ant Group (2022 - Present)
Tech Lead
- Led team of 5 engineers building payment reconciliation system
- Designed event-driven architecture using Kafka + Flink
- Strong expertise in JVM tuning, GC optimization

PROJECTS
Payment Reconciliation System | 2022 - 2024
- Tech Stack: Java, Spring Cloud, MySQL, Kafka, Flink, Redis
- Description: Handles 10B+ transactions/day
- Duration: 18 months

SKILLS
Java, Spring Cloud, MySQL, Kafka, Flink, Redis, JVM, Docker, Kubernetes,
distributed systems, system design

SELF SUMMARY
Senior backend engineer with 8 years of experience, Java ecosystem expert.
Limited Python experience, strong system design and leadership skills.
""",
        "ground_truth": {
            "backend_python_jd": 45,
            "frontend_vue_jd": 15,
            "algo_recommendation_jd": 40,
        },
    },
    # 8. 应届前端 (0 经验, 强 Vue)
    {
        "filename": "08_chen_shi_junior_frontend.pdf",
        "content": """Chen Shi
Email: chenshi@example.com  |  Phone: 138-0008-0008  |  Location: Wuhan

EDUCATION
Wuhan University, Bachelor of Software Engineering, 2022 - 2026 (expected)

INTERNSHIPS
- ByteDance Frontend Intern (Summer 2024): Built internal dashboards with Vue 3
- Alibaba Frontend Intern (Summer 2023): Worked on a React-based admin tool

PROJECTS
Personal Blog Platform | 2024
- Tech Stack: Vue 3, TypeScript, Node.js, MongoDB
- Description: Personal blog with admin panel, deployed on Vercel
- Duration: 2 months

Course Project: Mini React | 2023
- Tech Stack: JavaScript
- Description: Built a mini React (vDOM, diff algorithm) from scratch
- Duration: 1 month

SKILLS
Vue 3, TypeScript, JavaScript, HTML, CSS, React, Node.js, Git

SELF SUMMARY
Undergraduate student, frontend focused. Strong in Vue 3, learning React and
backend technologies.
""",
        "ground_truth": {
            "backend_python_jd": 20,
            "frontend_vue_jd": 70,
            "algo_recommendation_jd": 15,
        },
    },
    # 9. 数据工程师 (Python 强, 后端弱)
    {
        "filename": "09_yang_shy_data_eng.pdf",
        "content": """Yang Shi
Email: yangshi@example.com  |  Phone: 138-0009-0009  |  Location: Beijing

EDUCATION
Beihang University, Master of Data Science, 2019 - 2022
Beihang University, Bachelor of Statistics, 2015 - 2019

WORK EXPERIENCE
JD.com (2022 - Present)
Data Engineer
- Built ETL pipelines using Airflow + Spark + Python
- Maintained data warehouse on Snowflake (500TB+)
- Some machine learning model deployment using MLflow
- Wrote Python tooling for data quality monitoring

PROJECTS
Real-time User Behavior Analytics | 2023
- Tech Stack: Python, Spark, Kafka, Flink, ClickHouse
- Description: Real-time analytics pipeline processing 10B+ events/day
- Duration: 8 months

SKILLS
Python, Spark, Airflow, SQL, Snowflake, Kafka, ClickHouse, MLflow,
basic Docker, basic Kubernetes

SELF SUMMARY
Data engineer with 3 years of experience, strong in Python data stack.
Limited backend framework experience, strong in data pipelines and SQL.
""",
        "ground_truth": {
            "backend_python_jd": 55,
            "frontend_vue_jd": 20,
            "algo_recommendation_jd": 60,
        },
    },
    # 10. 弱匹配 (HR 实习生, 啥都不太行)
    {
        "filename": "10_xu_shiyong_weak.pdf",
        "content": """Xu Shi
Email: xushi@example.com  |  Phone: 138-0010-0010  |  Location: Changsha

EDUCATION
Hunan University, Bachelor of Business Administration, 2021 - 2025

WORK EXPERIENCE
Local Startup (2024 - Present, 6 months)
HR Intern
- Screened resumes manually
- Scheduled interviews
- Maintained Excel sheets

SKILLS
Excel, Word, basic PPT, English (CET-6)

SELF SUMMARY
Business school graduate with internship in HR. Limited technical skills,
strong communication and organization.
""",
        "ground_truth": {
            "backend_python_jd": 5,
            "frontend_vue_jd": 5,
            "algo_recommendation_jd": 5,
        },
    },
    # 11. 真实脱敏简历 (杜逸剑 - Java + Spring AI + 全栈, 在校大三)
    {
        "filename": "11_du_yijian_real_java.pdf",
        "content": """杜逸剑
Email: duyijian@example.com  |  Phone: 138-0011-0011  |  Location: Pingdingshan

EDUCATION
Pingdingshan University, Bachelor of Software Engineering, 2023.09 - 2027.06
Certificate: English CET-6

SKILLS
- Java: solid foundation; proficient in Spring Boot, MyBatis, Lombok; familiar with
  Axios and RESTful API design
- Spring AI: ChatClient, ChatMemory; familiar with Prompt design and Function Calling;
  hands-on experience integrating LLMs and streaming chat
- Familiar with Coze, Dify; experience building Agent workflows and custom plugins
- Familiar with Prompt Engineering, AI Agent, RAG; hands-on with AI Coding / LLM-assisted dev
- Redis: data structures, caching strategies, Redis Stream for async messaging, Lua
  atomic operations, well-versed in cache penetration/breakdown/avalanche issues
- Vue + Element UI; basic frontend development; WeChat mini-program experience
- Daily use of Cursor, Claude Code, Trae for vibecoding to debug and boost efficiency
- Proficient in Git, Markdown

PROJECTS
JK Smart Elder Care (Full-stack Developer) | 2025.12 - 2026.01
Tech: SpringAI + SpringBoot + MyBatis + MySQL + Redis + Vue 3 + WeChat Mini-Program
     + JWT + AliyunOSS + AOP + PageHelper + Nginx
- One-stop platform for community elder care: service packages, cart/checkout,
  WeChat Pay, address management, etc.
- LLM integration: extended AlibabaOpenAiChatModel, compat with Bailian, ensured
  Function Calling works
- AI tool calling + data trust: SpringAI @Tool queries DB before answering, suppresses
  service-name / price hallucination, implemented "AI helps user add to cart"
- Auth: JWT + custom interceptor, separate user-side and admin-side requests
- Cache: Redis hot service list + cache key version prefix for bulk invalidation
- Persistence: MyBatis + PageHelper for order / service / stats SQL
- Cross-cutting: SpringAOP for auto-filling common fields and decoupling logging

Food & Fun Local Platform (Full-stack Developer) | 2026.01 - 2026.03
Tech: SpringAI + Spring Boot + MyBatisPlus + MySQL + Redis + Nginx + Maven
     + ChatMemory + ChatClient
- One-stop local food / entertainment service + content platform
- AI chat: Spring AI ChatClient + OpenAI API; AOP-style Advisor chain for chat logging
  and memory management
- Async flash-sale: Redis Stream MQ, response 460ms -> 140ms
- Oversell prevention: Redisson distributed reentrant lock / optimistic lock
- Cache: multi-level Redis cache + Lua atomic ops; handles cache penetration/breakdown/avalanche
- Data safety: Redis master-slave + Sentinel cluster for dual-write consistency
- Deployment: Docker for services, Nginx for reverse proxy and load balancing

SELF SUMMARY
Deeply passionate about future-oriented tech (especially AI); follow and learn
the latest technologies and AI tools daily. Open to challenging work, lifelong
learner. Optimistic, detail-oriented, strong sense of responsibility; held multiple
department leader roles in college. Strong self-learning and problem-solving
ability; active on Bilibili, Gitee learning open source.
""",
        "ground_truth": {
            "backend_python_jd": 50,
            "frontend_vue_jd": 45,
            "algo_recommendation_jd": 30,
        },
    },
]


# ============================================================
# 3 个 JD
# ============================================================
JDS = {
    "backend_python_jd": """Senior Python Backend Engineer - Shanghai

We are looking for a senior Python backend engineer to join our platform team.

Requirements:
- 3+ years of Python development experience
- Strong knowledge of FastAPI or Django
- Experience with PostgreSQL and Redis
- Familiar with Docker and Kubernetes
- Experience with LangGraph or AutoGen is a strong plus
- Bachelor's degree in Computer Science or related field

Nice to have:
- Open source contributions
- Experience with multi-agent systems
- Experience leading a small team

Responsibilities:
- Design and implement RESTful APIs
- Mentor junior engineers
- Participate in architecture decisions

Salary: 30K-50K RMB/month
""",
    "frontend_vue_jd": """Senior Vue 3 Frontend Engineer - Hangzhou

We are looking for a senior Vue 3 frontend engineer to join our product team.

Requirements:
- 3+ years of Vue.js development experience, with at least 1 year on Vue 3
- Strong knowledge of TypeScript and modern JavaScript (ES6+)
- Experience with state management (Pinia or Vuex)
- Familiar with build tools (Vite or Webpack)
- Experience with component library design
- Bachelor's degree in Computer Science or related field

Nice to have:
- React experience
- ECharts or data visualization experience
- Performance optimization experience

Responsibilities:
- Build and maintain large-scale SPAs
- Design reusable component libraries
- Mentor junior engineers
- Collaborate with backend team on API design

Salary: 25K-45K RMB/month
""",
    "algo_recommendation_jd": """Senior Recommendation Algorithm Engineer - Beijing

We are looking for a senior recommendation algorithm engineer to join our content team.

Requirements:
- 3+ years of experience in recommendation systems or search ranking
- Strong knowledge of DNN, multi-task learning, and embedding models
- Experience with PyTorch or TensorFlow
- Familiar with A/B testing and offline metrics (AUC, NDCG, CTR)
- Master's degree or above in CS, Math, or related field
- Publications at RecSys / SIGIR / KDD is a strong plus

Nice to have:
- Experience with large-scale feature stores
- Multi-modal learning experience
- C++ / CUDA for performance-critical code

Responsibilities:
- Design and deploy recommendation models
- Lead A/B testing and drive metric improvements
- Collaborate with engineering team on system design
- Publish papers and contribute to the community

Salary: 40K-70K RMB/month
""",
}


def write_resume_pdf(filename: str, content: str) -> None:
    """用 PyMuPDF 写一份 PDF."""
    path = RESUMES_DIR / filename
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    y = 72
    for line in content.splitlines():
        if y > 800:  # 超出一页则加新页
            page = doc.new_page(width=595, height=842)
            y = 72
        if line.strip():
            page.insert_text((50, y), line, fontsize=10)
        y += 13
    doc.save(path)
    doc.close()


def write_jd_txt(filename: str, content: str) -> None:
    """JD 用 txt 存 (后端支持)."""
    (JDS_DIR / filename).write_text(content, encoding="utf-8")


def main() -> None:
    print(f"Generating {len(RESUMES)} resumes and {len(JDS)} JDs...")
    for r in RESUMES:
        write_resume_pdf(r["filename"], r["content"])
        print(f"  resume: {r['filename']}")
    for jd_name, jd_text in JDS.items():
        write_jd_txt(f"{jd_name}.txt", jd_text)
        print(f"  jd: {jd_name}.txt")

    # 写 ground truth JSON
    gt = {r["filename"]: r["ground_truth"] for r in RESUMES}
    (DATA_DIR / "ground_truth.json").write_text(
        json.dumps(gt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
        print(f"  ground_truth.json ({len(gt)} entries)")


if __name__ == "__main__":
    main()
