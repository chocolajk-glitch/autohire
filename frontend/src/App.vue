<script setup>
import { ref, computed, nextTick, watch, onMounted } from 'vue'
import { JD_LABELS, RESUME_LABELS } from './config.js'

// API base: 开发环境用 Vite 代理 (相对路径 /api), 生产环境可指向实际后端域名
const API = ''

// 状态
const jds = ref([])
const resumes = ref([])
const selectedJD = ref('')
const selectedResumes = ref([])
const enableReflection = ref(false)
const runQuestions = ref(false)
const llmProvider = ref('minimax')

// 模型选项
const LLM_OPTIONS = [
  { value: 'minimax', label: 'MiniMax M2.7', desc: 'minimaxi.com · OpenAI 兼容 · 速度最快 · 已配置 API Key' },
  { value: 'qwen', label: '通义千问 Qwen', desc: '阿里云百炼 · 中文强 · 余额已欠费 (暂时不可用)' },
  { value: 'deepseek', label: 'DeepSeek', desc: '深度求索 · 推理强 · 余额已欠费 (暂时不可用)' },
]

const jobId = ref(null)
const isRunning = ref(false)
const progress = ref(0)
const currentCandidate = ref('')
const logs = ref([])
const result = ref(null)
const summary = ref({})
const ranking = ref([])
const candidates = ref([])
const tab = ref('ranking')
const logBox = ref(null)

// 5 阶段进度
const STAGE_DEFS = [
  { key: 'parse_jd', label: '解析 JD', icon: '📄' },
  { key: 'parse_resume', label: '解析简历', icon: '📋' },
  { key: 'match', label: '匹配 + 反思', icon: '🎯' },
  { key: 'interview_questions_crew', label: '生成面试题', icon: '💬' },
  { key: 'generate_report', label: '生成报告', icon: '📊' },
]
const stages = ref(STAGE_DEFS.map(s => ({ ...s, status: 'pending', duration_ms: 0 })))

// 动态路由信息
const routeInfo = ref(null)
const ROUTE_LABELS = {
  algorithm_specialist: '算法专项匹配',
  frontend_specialist: '前端专项匹配',
  ocr_fallback: 'OCR 解析路径',
  standard: '标准匹配路径',
}
const routeLabel = computed(() => routeInfo.value ? (ROUTE_LABELS[routeInfo.value.route] || routeInfo.value.route) : '')

// 计算属性
const canStart = computed(() => selectedJD.value && selectedResumes.value.length > 0 && !isRunning.value)
const statusText = computed(() => {
  if (isRunning.value) return '运行中'
  if (result.value) return '已完成'
  return '空闲'
})
const statusBadgeClass = computed(() => {
  if (isRunning.value) return 'warn'
  if (result.value) return 'ok'
  return 'info'
})
const resumeOptions = computed(() => resumes.value.map(r => ({ key: r, label: RESUME_LABELS[r] || r })))
const currentCandidateLabel = computed(() => {
  if (!currentCandidate.value) return ''
  return RESUME_LABELS[currentCandidate.value] || currentCandidate.value
})
const stageProgressWidth = computed(() => {
  const done = stages.value.filter(s => s.status === 'success' || s.status === 'failed').length
  const running = stages.value.find(s => s.status === 'running')
  const N = stages.value.length
  const segmentPct = 100 / (N - 1)
  let base = done * segmentPct
  if (running) base += segmentPct * 0.5
  return Math.min(100, base)
})
const currentStageMessage = computed(() => {
  const running = stages.value.find(s => s.status === 'running')
  if (!running) {
    if (result.value) {
      const totalSec = (summary.value.duration_seconds || 0).toFixed(1)
      return `✅ 全部完成 · 总耗时 ${totalSec} 秒`
    }
    return ''
  }
  const cand = currentCandidateLabel.value
  const idx = stages.value.findIndex(s => s.status === 'running')
  const cur = stages.value[idx]
  const next = stages.value[idx + 1]
  let msg = `正在执行: ${cur.icon} ${cur.label}`
  if (cand) msg += ` · 当前候选人: ${cand}`
  if (next) msg += ` · 下一阶段: ${next.icon} ${next.label}`
  return msg
})

// 工具函数
function getJDLabel(key) { return JD_LABELS[key] || key }
function formatTime(ts) { return new Date(ts * 1000).toLocaleTimeString('zh-CN') }
function formatLogMsg(l) {
  if (l.event === 'processing') {
    const name = RESUME_LABELS[l.candidate] || l.candidate
    return `正在评估 ${name}（第 ${l.idx} / ${l.total} 份）`
  }
  if (l.event === 'done') return `✅ 全部完成 · 平均分 ${(l.avg_score || 0).toFixed(1)} · 成功 ${l.succeeded} 份 · 失败 ${l.failed || 0} 份`
  if (l.event === 'hitl_submitted') return `⚠️  已自动提交 ${l.count} 份需要 HR 复核的候选人`
  if (l.event === 'job_created') return `任务创建 · 共 ${l.total} 份简历待评估`
  if (l.event === 'started') return '开始处理'
  if (l.event === 'error') return `❌ 错误: ${l.msg}`
  return ''
}
function recText(r) {
  return { strong_recommend: '强烈推荐', recommend: '推荐', neutral: '中性', not_recommend: '不推荐' }[r] || r
}
function recConf(c) { return { high: '高置信度', medium: '中置信度', low: '低置信度' }[c] || c }
function getRankClass(i) { return i === 0 ? 'top1' : i === 1 ? 'top2' : i === 2 ? 'top3' : '' }
function getCandidateByName(name) { return candidates.value.find(c => c.candidate_name === name) }
function stageIcon(s) {
  if (s.status === 'success') return '✅'
  if (s.status === 'failed') return '❌'
  if (s.status === 'running') return s.icon
  return '○'
}
function stageTime(s) {
  if (s.duration_ms > 0) return `${(s.duration_ms / 1000).toFixed(1)}s`
  if (s.status === 'pending') return '待执行'
  if (s.status === 'running') return '进行中...'
  return ''
}
function resetStages() {
  stages.value = STAGE_DEFS.map(s => ({ ...s, status: 'pending', duration_ms: 0 }))
}
function _applyStageEvent(stepName, status, durationMs = 0) {
  let key = stepName
  if (key === 'match_with_reflection' || key === 'match') key = 'match'
  const idx = stages.value.findIndex(s => s.key === key)
  if (idx < 0) return
  const cur = stages.value[idx]
  if (cur.status === 'running' && durationMs > 0) cur.duration_ms = durationMs
  cur.status = status
}

// 用户操作
function selectAll() { selectedResumes.value = [...resumes.value] }
function selectNone() { selectedResumes.value = [] }
function selectSmart() {
  const strong = resumes.value.find(r => r.includes('strong'))
  const mid = resumes.value.find(r => r.includes('fullstack') || r.includes('data_eng') || r.includes('real_java'))
  const weak = resumes.value.find(r => r.includes('weak'))
  selectedResumes.value = [strong, mid, weak].filter(Boolean)
}

async function loadData() {
  const [jdsR, resumesR] = await Promise.all([
    fetch(`${API}/api/jd`).then(r => r.json()),
    fetch(`${API}/api/resumes`).then(r => r.json()),
  ])
  jds.value = jdsR.map(k => ({ key: k, label: JD_LABELS[k] || k }))
  resumes.value = resumesR
  if (jds.value.length > 0 && !selectedJD.value) selectedJD.value = jds.value[0].key
}

// 上传相关
const uploading = ref(false)
const uploadMsg = ref('')
const jdFileInput = ref(null)
const resumeFileInput = ref(null)

async function uploadJD(e) {
  const file = e.target.files?.[0]
  if (!file) return
  uploading.value = true
  uploadMsg.value = `正在上传 ${file.name}...`
  try {
    const form = new FormData()
    form.append('file', file)
    const resp = await fetch(`${API}/api/jd/upload`, { method: 'POST', body: form })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }))
      throw new Error(err.detail || '上传失败')
    }
    const data = await resp.json()
    uploadMsg.value = `✓ 已上传: ${data.filename}（${data.job_title || '解析完成'}）`
    // 刷新 JD 列表并选中新上传的
    await loadData()
    const newKey = file.name.replace(/\.[^.]+$/, '')
    selectedJD.value = newKey
  } catch (err) {
    uploadMsg.value = `✗ 上传失败: ${err.message}`
  } finally {
    uploading.value = false
    if (jdFileInput.value) jdFileInput.value.value = ''
  }
}

async function uploadResume(e) {
  const files = e.target.files
  if (!files?.length) return
  uploading.value = true
  uploadMsg.value = `正在上传 ${files.length} 份简历...`
  let ok = 0, fail = 0
  for (const file of files) {
    try {
      const form = new FormData()
      form.append('file', file)
      const resp = await fetch(`${API}/api/resume/upload`, { method: 'POST', body: form })
      if (!resp.ok) fail++
      else ok++
    } catch {
      fail++
    }
  }
  uploadMsg.value = `✓ 上传完成: 成功 ${ok} 份` + (fail > 0 ? `, 失败 ${fail} 份` : '')
  await loadData()
  uploading.value = false
  if (resumeFileInput.value) resumeFileInput.value.value = ''
}

async function startBatch() {
  logs.value = []
  result.value = null
  progress.value = 0
  currentCandidate.value = ''
  resetStages()
  routeInfo.value = null
  isRunning.value = true

  const resp = await fetch(`${API}/api/batch/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jd_filename: selectedJD.value,
      resume_filenames: selectedResumes.value,
      enable_reflection: enableReflection.value,
      run_interview_questions: runQuestions.value,
      llm_provider: llmProvider.value,
      auto_submit_hitl: true,
    }),
  })
  const data = await resp.json()
  jobId.value = data.job_id
  listenSSE(data.job_id)
}

function listenSSE(jid) {
  const es = new EventSource(`${API}/api/batch/${jid}/stream`)
const handler = (eventName) => (e) => {
            const data = JSON.parse(e.data)
            if (eventName === 'log') {
              if (data.event === 'stage') {
                _applyStageEvent(data.step, data.status, data.duration_ms || 0)
                // 路由决策事件 (后端推 'route_detected' stage)
                if (data.step === 'route_detected' && data.route) {
                  routeInfo.value = {
                    route: data.route,
                    reason: data.reason || '',
                  }
                }
                return
              }
              logs.value.push(data)
              nextTick(() => { if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight; })
            } else if (eventName === 'progress') {
      progress.value = data.progress
      currentCandidate.value = data.current_candidate
    } else if (eventName === 'status') {
      if (data.summary) summary.value = data.summary
    } else if (eventName === 'result') {
      summary.value = data.summary
      ranking.value = data.ranking
      candidates.value = data.candidates
      result.value = data
    } else if (eventName === 'end') {
      isRunning.value = false
      es.close()
    }
  }
  es.addEventListener('status', handler('status'))
  es.addEventListener('log', handler('log'))
  es.addEventListener('progress', handler('progress'))
  es.addEventListener('result', handler('result'))
  es.addEventListener('end', handler('end'))
}

watch(logs, () => {
  nextTick(() => { if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight; })
})

onMounted(loadData)
</script>

<template>
  <div class="container">
    <header>
      <div class="header-row">
        <div class="logo">
          <span class="dot"></span>
          <span>AutoHire</span>
        </div>
        <h1>多 Agent 智能招聘筛选系统</h1>
      </div>
      <div class="subtitle">由 AutoGen + CrewAI 驱动 · 支持通义千问 / MiniMax / DeepSeek · 含规划调度 · 自我反思 · 人机协同</div>
    </header>

    <div class="grid">
      <!-- 左侧: 配置 + 控制 -->
      <div>
        <div class="card">
          <h2>① 选择要招聘的岗位（JD）</h2>
          <select v-model="selectedJD">
            <option value="">-- 请选择岗位 --</option>
            <option v-for="jd in jds" :key="jd.key" :value="jd.key">{{ jd.label }}</option>
          </select>
          <div class="hint" v-if="selectedJD">
            ✓ 已选岗位: <strong>{{ getJDLabel(selectedJD) }}</strong><br>
            <span style="color:#94a3b8;">文件: data/jds/{{ selectedJD }}.txt</span>
          </div>
          <div class="hint" v-else>岗位文件存放在 data/jds/ 目录，可在下方操作面板中查看内容</div>
          <div class="upload-zone" @click="jdFileInput?.click()" :class="{ disabled: uploading }">
            <input ref="jdFileInput" type="file" accept=".txt,.pdf,.docx,.md" hidden @change="uploadJD" />
            <span class="upload-icon">📤</span>
            <span>点击上传 JD 文件（TXT / PDF / DOCX）</span>
          </div>
        </div>

        <div class="card" style="margin-top: 16px;">
          <h2>② 选择要评估的简历（{{ selectedResumes.length }} / {{ resumes.length }} 份）</h2>
          <div class="resume-list">
            <label v-for="r in resumeOptions" :key="r.key" class="resume-item">
              <input type="checkbox" :value="r.key" v-model="selectedResumes" />
              <span class="name">{{ r.label }}</span>
            </label>
          </div>
          <div style="display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap;">
            <button style="width: auto; padding: 6px 12px; font-size: 12px; background: #334155;" @click="selectAll">全选</button>
            <button style="width: auto; padding: 6px 12px; font-size: 12px; background: #334155;" @click="selectNone">清空</button>
            <button style="width: auto; padding: 6px 12px; font-size: 12px; background: #334155;" @click="selectSmart" title="自动选 1 份强匹配 + 1 份中等 + 1 份弱匹配用于演示">快速演示 (3 份)</button>
          </div>
          <div class="upload-zone" @click="resumeFileInput?.click()" :class="{ disabled: uploading }">
            <input ref="resumeFileInput" type="file" accept=".pdf,.docx,.txt,.md" multiple hidden @change="uploadResume" />
            <span class="upload-icon">📤</span>
            <span>点击上传简历（支持多选，PDF / DOCX / TXT）</span>
          </div>
          <div class="hint">📁 简历来源: data/resumes/ 目录，共 {{ resumes.length }} 份</div>
        </div>

        <div class="card" style="margin-top: 16px;">
          <h2>③ 评估选项（高级）</h2>
          <label style="display: flex; align-items: flex-start; gap: 8px; font-size: 13px; color: #cbd5e1;">
            <input type="checkbox" v-model="enableReflection" style="margin-top: 2px;" />
            <div>
              <div>启用自我反思（让 LLM 自查漏判）</div>
              <div class="hint">开启后匹配度更高，但每份简历多花约 30 秒</div>
            </div>
          </label>
          <label style="display: flex; align-items: flex-start; gap: 8px; font-size: 13px; color: #cbd5e1; margin-top: 10px;">
            <input type="checkbox" v-model="runQuestions" style="margin-top: 2px;" />
            <div>
              <div>自动生成定制化面试题（CrewAI 协作）</div>
              <div class="hint">开启后调用 CrewAI 三角色（研究员/出题人/审核员）出题，每份多花约 70 秒</div>
            </div>
          </label>
          <div style="margin-top: 12px;">
            <label style="font-size: 13px; color: #cbd5e1; display: block; margin-bottom: 6px;">
              🤖 选择大模型
            </label>
            <select v-model="llmProvider">
              <option v-for="opt in LLM_OPTIONS" :key="opt.value" :value="opt.value">
                {{ opt.label }}
              </option>
            </select>
            <div class="hint" style="margin-top: 4px;">
              {{ (LLM_OPTIONS.find(o => o.value === llmProvider) || {}).desc }}
            </div>
          </div>
        </div>

        <div class="card" style="margin-top: 16px;">
          <button :disabled="!canStart" @click="startBatch">
            {{ isRunning ? '⏳ 正在评估中...' : '🚀 开始批量评估' }}
          </button>
          <div v-if="uploadMsg" class="upload-status" :class="{ error: uploadMsg.startsWith('✗') }">
            {{ uploadMsg }}
          </div>
          <div v-if="jobId" class="progress-wrap">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: (progress * 100) + '%' }"></div>
            </div>
            <div class="progress-text">
              总体进度 {{ (progress * 100).toFixed(0) }}% · 当前: {{ currentCandidateLabel || '准备中' }}
            </div>
          </div>
          <div v-if="!canStart && !isRunning" class="hint" style="margin-top: 8px;">
            ⚠️ 请先选岗位和至少 1 份简历
          </div>
        </div>
      </div>

      <!-- 右侧: 实时日志 + 结果 -->
      <div>
        <div class="stage-card">
          <h2>
            评估进度
            <span v-if="isRunning" class="badge running">运行中</span>
            <span v-else-if="result" class="badge ok">已完成</span>
            <span v-else class="badge info">待开始</span>
          </h2>
          <div class="stage-bar">
            <div class="stage-fill" :style="{ width: stageProgressWidth + '%' }"></div>
            <div v-for="s in stages" :key="s.key" class="stage-item">
              <div class="stage-dot" :class="s.status">{{ stageIcon(s) }}</div>
              <div class="stage-label">{{ s.label }}</div>
              <div class="stage-time">{{ stageTime(s) }}</div>
            </div>
          </div>
          <div v-if="currentStageMessage" class="stage-current">
            <span style="font-size: 16px;">⏳</span>
            <span>{{ currentStageMessage }}</span>
          </div>
          <div v-if="routeInfo" class="stage-current" style="background: var(--surface); border-style: dashed; margin-top: 8px;">
            <span style="font-size: 14px;">🧭</span>
            <span>动态路由: <strong>{{ routeLabel }}</strong> · {{ routeInfo.reason }}</span>
          </div>
        </div>

        <div class="card">
          <h2>
            实时运行日志
            <span class="badge" :class="statusBadgeClass">{{ statusText }}</span>
          </h2>
          <div class="log" ref="logBox">
            <div v-for="(l, i) in logs" :key="i">
              <span class="ts">{{ formatTime(l.ts) }}</span>
              <span class="ev"> [{{ l.event }}]</span>
              <span class="msg"> {{ formatLogMsg(l) }}</span>
            </div>
            <div v-if="!logs.length && !isRunning" class="empty">
              还没有日志，点左侧"开始批量评估"即可看到 Agent 实时处理过程
            </div>
          </div>
        </div>

        <div class="card" style="margin-top: 16px;" v-if="result">
          <h2>📊 评估结果</h2>
          <div class="stat-grid">
            <div class="stat">
              <div class="num">{{ summary.succeeded }}</div>
              <div class="lbl">成功评估</div>
            </div>
            <div class="stat">
              <div class="num">{{ summary.avg_score }}</div>
              <div class="lbl">平均分</div>
            </div>
            <div class="stat">
              <div class="num">{{ summary.median_score }}</div>
              <div class="lbl">中位数分</div>
            </div>
            <div class="stat">
              <div class="num" :style="{ color: summary.hitl_count > 0 ? '#fcd34d' : '#60a5fa' }">{{ summary.hitl_count }}</div>
              <div class="lbl">需人工复核</div>
            </div>
          </div>

          <div class="tab-bar">
            <div class="tab" :class="{ active: tab === 'ranking' }" @click="tab = 'ranking'">🏆 排行榜</div>
            <div class="tab" :class="{ active: tab === 'cards' }" @click="tab = 'cards'">📋 详情卡片</div>
          </div>

          <div v-if="tab === 'ranking'" class="ranking">
            <div v-for="(name, i) in ranking" :key="name" class="rank-row" :class="getRankClass(i)">
              <div class="pos" :class="{ top: i === 0 }">{{ i + 1 }}</div>
              <div class="name">
                {{ name }}
                <span v-if="getCandidateByName(name)?.needs_human_review" class="hitl-tag">需人工复核</span>
              </div>
              <div class="score">{{ getCandidateByName(name)?.match.overall_score ?? '-' }}</div>
              <div class="rec" :class="getCandidateByName(name)?.recommendation">
                {{ recText(getCandidateByName(name)?.recommendation) }}
              </div>
              <div class="conf">
                {{ recConf(getCandidateByName(name)?.match.confidence) }}
              </div>
            </div>
          </div>

          <div v-if="tab === 'cards'">
            <div v-for="c in candidates" :key="c.candidate_name" class="cand-card" :class="{ hitl: c.needs_human_review }">
              <div class="head">
                <div class="name">
                  {{ c.candidate_name }}
                  <span v-if="c.needs_human_review" class="hitl-tag">需复核</span>
                </div>
                <div class="rec" :class="c.recommendation">{{ recText(c.recommendation) }}</div>
              </div>
              <div class="reason">
                <strong>综合分 {{ c.match.overall_score }}</strong>
                · {{ c.recommendation_reason }}
              </div>
              <details>
                <summary>查看优势 / 不足 / 反思</summary>
                <div style="margin-top: 10px; line-height: 1.7;">
                  <div><span class="strengths">✓ 优势</span> {{ (c.match.strengths || []).join('; ') }}</div>
                  <div style="margin-top: 4px;"><span class="weaknesses">✗ 不足</span> {{ (c.match.weaknesses || []).join('; ') }}</div>
                  <div v-if="c.match.reflection_note" style="margin-top: 4px;"><span class="reflection">⟳ 反思</span> {{ c.match.reflection_note }}</div>
                </div>
              </details>
            </div>
          </div>
        </div>

        <div v-else-if="!isRunning && !result" class="card" style="text-align: center; padding: 60px 20px;">
          <div class="empty">
            👈 在左侧选好岗位和简历后<br>点 "🚀 开始批量评估" 即可开始
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
