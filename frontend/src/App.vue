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
const runQuestions = ref(false)
const useAutoGen = ref(false)
const llmProvider = ref('minimax')

// 模型选项
const LLM_OPTIONS = [
  { value: 'minimax', label: 'MiniMax M2.7', desc: 'minimaxi.com · OpenAI 兼容 · 速度最快 · 已配置 API Key' },
  { value: 'qwen', label: '通义千问 Qwen', desc: '阿里云百炼 · 中文强' },
  { value: 'deepseek', label: 'DeepSeek', desc: '深度求索 · 推理强' },
]                   

const jobId = ref(null)
const isRunning = ref(false)
const progress = ref(0)
const currentCandidate = ref('')
const logs = ref([])
// 并发模式下: 每个候选人独立的进度
// 结构: { [resume_idx]: { name, status, currentStep, stageStates } }
const candidateProgress = ref({})
const result = ref(null)
const summary = ref({})
const ranking = ref([])
const candidates = ref([])
const tab = ref('ranking')
const logBox = ref(null)

// 5 阶段进度
const STAGE_DEFS = [
  {
    key: 'parse',
    label: '解析 JD + 简历',
    icon: '📑',
    children: [
      { key: 'parse_jd', label: '解析 JD' },
      { key: 'parse_resume', label: '解析简历' },
    ],
  },
  { key: 'match', label: '匹配 + 反思', icon: '🎯' },
  { key: 'interview_questions_crew', label: '生成面试题', icon: '💬' },
  { key: 'generate_report', label: '生成报告', icon: '📊' },
]
const stages = ref(STAGE_DEFS.map(s => ({
  ...s,
  status: 'pending',
  duration_ms: 0,
  child_status: s.children ? Object.fromEntries(s.children.map(c => [c.key, 'pending'])) : null,
})))

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
  if (l.event === 'stage' && l.status === 'failed') {
    const cand = l.resume_name ? ` [${RESUME_LABELS[l.resume_name] || l.resume_name}]` : ''
    return `⚠️  ${l.step} 失败${cand}${l.error ? `: ${l.error}` : ''}`
  }
  return ''
}
function recText(r) {
  return { strong_recommend: '强烈推荐', recommend: '推荐', neutral: '中性', not_recommend: '不推荐' }[r] || r
}
function recConf(c) { return { high: '高置信度', medium: '中置信度', low: '低置信度' }[c] || c }
function questionCategory(c) {
  return { technical: '技术', project: '项目', behavioral: '行为', system_design: '系统设计', coding: '编程', other: '其他' }[c] || c
}
function questionDifficulty(d) {
  return { easy: '简单', medium: '中等', hard: '困难' }[d] || d
}
function getRankClass(i) { return i === 0 ? 'top1' : i === 1 ? 'top2' : i === 2 ? 'top3' : '' }
function getCandidateByName(name) { return candidates.value.find(c => c.candidate_name === name) }
function turnAvatar(source) {
  if (source === 'Assessor') return '🧑‍⚖️'
  if (source === 'Refiner') return '🔍'
  if (source === 'SelectorGroupChat') return '🎯'
  return '💬'
}
function turnTypeLabel(type) {
  if (type === 'TextMessage') return '发言'
  if (type === 'SelectSpeakerEvent') return '选人'
  if (type === 'SelectorEvent') return '决策推理'
  return type
}
function stageIcon(s) {
  if (s.status === 'success') return '✓'
  if (s.status === 'failed') return '✗'
  if (s.status === 'running') return '⏳'
  return '○'
}
function resumeLabel(name) {
  if (!name) return ''
  return RESUME_LABELS[name] || RESUME_LABELS[name.replace(/\.pdf$/, '')] || name
}
function stageShortLabel(k) {
  const map = { parse: '解析', match: '匹配', generate_questions: '出题', generate_report: '报告' }
  return map[k] || k
}
function stageTime(s) {
  if (s.duration_ms > 0) return `${(s.duration_ms / 1000).toFixed(1)}s`
  if (s.status === 'pending') return '待执行'
  if (s.status === 'running') return '进行中...'
  return ''
}
function resetStages() {
  stages.value = STAGE_DEFS.map(s => ({
    ...s,
    status: 'pending',
    duration_ms: 0,
    child_status: s.children ? Object.fromEntries(s.children.map(c => [c.key, 'pending'])) : null,
  }))
  candidateProgress.value = {}
}

function _onProcessingEvent(data) {
  // data: { event: 'processing', status: 'running'|'success'|'failed', resume_idx, resume_name, total }
  const idx = data.resume_idx
  if (idx === null || idx === undefined) return
  if (!candidateProgress.value[idx]) {
    candidateProgress.value[idx] = {
      name: data.resume_name,
      status: 'pending',
      stages: {
        parse: { status: 'pending', duration_ms: 0, error: null },
        match: { status: 'pending', duration_ms: 0, error: null },
        generate_questions: { status: 'pending', duration_ms: 0, error: null },
        generate_report: { status: 'pending', duration_ms: 0, error: null },
      },
    }
  }
  candidateProgress.value[idx].name = data.resume_name
  candidateProgress.value[idx].status = data.status
  candidateProgress.value = { ...candidateProgress.value }
}
function _applyStageEvent(stepName, status, durationMs = 0, resumeIdx = null, resumeName = null, errorMsg = null) {
  // 并发模式: 更新该候选人自己的 stage 状态 (而不影响全局 stages)
  if (resumeIdx !== null && resumeName) {
    _updateCandidateProgress(resumeIdx, resumeName, stepName, status, durationMs, errorMsg)
    // match key 别名
    let key = stepName
    if (key === 'match_with_reflection' || key === 'match' || key === 'autogen_matcher_team') key = 'match'
    // 全局 stages 仍按 "当前活动候选人" 显示 (取最新 active 的)
    if (status === 'running') {
      _applyGlobalStage(key, status, durationMs)
    } else if (status === 'success' || status === 'failed') {
      // 仅当该候选人所有步骤都完成, 才更新全局 stages 为完成态
      // 简化: 总是更新全局 stages 为最终态, 前端用 candidateProgress 主导显示
      _applyGlobalStage(key, status, durationMs)
    }
    return
  }

  // 串行模式: 单候选人, 直接更新全局 stages
  _applyGlobalStage(stepName, status, durationMs)
}

function _applyGlobalStage(stepName, status, durationMs) {
  let key = stepName
  if (key === 'match_with_reflection' || key === 'match' || key === 'autogen_matcher_team') key = 'match'

  // 先找是否是某个 stage 的 child
  for (const stage of stages.value) {
    if (stage.children) {
      const child = stage.children.find(c => c.key === key)
      if (child) {
        stage.child_status[key] = status
        if (status === 'success' && durationMs > 0) {
          stage.duration_ms = Math.max(stage.duration_ms || 0, durationMs)
        }
        if (status === 'running') {
          stage.status = 'running'
        }
        if (Object.values(stage.child_status).every(s => s === 'success')) {
          stage.status = 'success'
        } else if (Object.values(stage.child_status).some(s => s === 'failed')) {
          stage.status = 'failed'
        }
        return
      }
    }
  }

  const idx = stages.value.findIndex(s => s.key === key)
  if (idx < 0) return
  const cur = stages.value[idx]
  if (cur.status === 'running' && durationMs > 0) cur.duration_ms = durationMs
  cur.status = status
}

function _updateCandidateProgress(idx, name, stepName, status, durationMs, errorMsg = null) {
  // 初始化该候选人
  if (!candidateProgress.value[idx]) {
    candidateProgress.value[idx] = {
      name,
      status: 'pending',
      stages: {
        parse: { status: 'pending', duration_ms: 0, error: null },
        match: { status: 'pending', duration_ms: 0, error: null },
        generate_questions: { status: 'pending', duration_ms: 0, error: null },
        generate_report: { status: 'pending', duration_ms: 0, error: null },
      },
    }
  }
  const cp = candidateProgress.value[idx]
  cp.name = name

  // stepName -> 候选人自己的 stage key
  let stageKey = stepName
  if (stepName === 'parse_jd' || stepName === 'parse_resume' || stepName === 'route_detected') stageKey = 'parse'
  else if (stepName === 'match_with_reflection' || stepName === 'match' || stepName === 'autogen_matcher_team') stageKey = 'match'
  else if (stepName === 'interview_questions_crew') stageKey = 'generate_questions'

  if (cp.stages[stageKey]) {
    cp.stages[stageKey].status = status
    if (status === 'success' && durationMs > 0) {
      cp.stages[stageKey].duration_ms = durationMs
    }
    if (status === 'failed' && errorMsg) {
      cp.stages[stageKey].error = errorMsg
    }
  }

  // 候选人整体状态
  const allStages = Object.values(cp.stages)
  if (allStages.every(s => s.status === 'success')) {
    cp.status = 'success'
  } else if (allStages.some(s => s.status === 'failed')) {
    cp.status = 'failed'
  } else if (allStages.some(s => s.status === 'running')) {
    cp.status = 'running'
  }

  // 强制触发响应式更新
  candidateProgress.value = { ...candidateProgress.value }
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
      run_interview_questions: runQuestions.value,
      use_autogen: useAutoGen.value,
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
                _applyStageEvent(
                  data.step, data.status, data.duration_ms || 0,
                  data.resume_idx, data.resume_name, data.error,
                )
                // 路由决策事件 (后端推 'route_detected' stage)
                if (data.step === 'route_detected' && data.route) {
                  routeInfo.value = {
                    route: data.route,
                    reason: data.reason || '',
                  }
                }
                // 反思对话 (后端推 'autogen_matcher_team' success, 附加 messages)
                if (data.step === 'autogen_matcher_team' && data.messages && data.resume_name) {
                  const c = candidates.value.find(x => x.candidate_name === data.resume_name)
                  if (c) {
                    c.reflection_messages = data.messages
                    candidates.value = [...candidates.value]  // 触发响应式刷新
                  }
                }
                return
              }
              if (data.event === 'processing') {
                _onProcessingEvent(data)
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

// HR 复核
const hitlTab = ref(false)
const hitlPending = ref([])
const hitlLoading = ref(false)
const hitlSubmitting = ref({})
const hitlResult = ref({})  // { [candidate_name]: 'ok' | 'error' }

async function loadHitlPending() {
  hitlLoading.value = true
  try {
    const resp = await fetch(`${API}/api/hitl/pending`)
    hitlPending.value = await resp.json()
  } catch (e) {
    hitlPending.value = []
  } finally {
    hitlLoading.value = false
  }
}

async function submitHitlDecision(item) {
  const key = item.candidate_name
  hitlSubmitting.value[key] = true
  hitlResult.value[key] = null
  try {
    const body = {
      candidate_name: item.candidate_name,
      job_title: item.job_title,
    }
    if (item._adjusted_score !== undefined && item._adjusted_score !== '') {
      body.adjusted_score = Number(item._adjusted_score)
    }
    if (item._recommendation) {
      body.recommendation = item._recommendation
    }
    if (item._note) {
      body.note = item._note
    }
    const resp = await fetch(`${API}/api/hitl/decide`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: '提交失败' }))
      hitlResult.value[key] = 'error'
      alert(err.detail || '提交失败')
    } else {
      hitlResult.value[key] = 'ok'
      // 从列表中移除
      hitlPending.value = hitlPending.value.filter(p => p.candidate_name !== key)
    }
  } catch {
    hitlResult.value[key] = 'error'
  } finally {
    hitlSubmitting.value[key] = false
  }
}

function toggleHitlTab() {
  hitlTab.value = !hitlTab.value
  if (hitlTab.value) loadHitlPending()
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
            <input type="checkbox" v-model="useAutoGen" style="margin-top: 2px;" />
            <div>
              <div>启用 AutoGen 反思机制 <span style="background: #7c3aed; color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 11px; margin-left: 4px;">推荐</span></div>
              <div class="hint">Matcher 用 AutoGen SelectorGroupChat：Assessor 初评 + Refiner 审查的双 Agent 协作反思，发现漏判/等价技能（如 Flask ≈ Web 框架）</div>
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
              <!-- 子步骤状态 (用于并行解析) -->
              <div v-if="s.children && (s.status === 'running' || s.status === 'success' || s.status === 'failed')" class="stage-children">
                <div v-for="c in s.children" :key="c.key" class="stage-child">
                  <span :class="['child-dot', s.child_status?.[c.key]]">
                    {{ s.child_status?.[c.key] === 'success' ? '✓' : s.child_status?.[c.key] === 'failed' ? '✗' : '○' }}
                  </span>
                  <span :class="{ 'child-done': s.child_status?.[c.key] === 'success' }">{{ c.label }}</span>
                </div>
              </div>
            </div>
          </div>

          <!-- 并发模式: 多候选人分组进度 -->
          <div v-if="Object.keys(candidateProgress).length > 1" class="candidate-list">
            <div v-for="(cp, idx) in candidateProgress" :key="idx" class="candidate-row">
              <div class="candidate-name">
                <span :class="['candidate-dot', cp.status]">
                  {{ cp.status === 'success' ? '✓' : cp.status === 'failed' ? '✗' : cp.status === 'running' ? '⏳' : '○' }}
                </span>
                <span>{{ resumeLabel(cp.name) }}</span>
              </div>
              <div class="candidate-stages">
                <span v-for="(s, k) in cp.stages" :key="k" :class="['mini-stage', s.status]" :title="s.error ? (stageShortLabel(k) + ': ' + s.error) : (k + ': ' + s.status)">
                  {{ stageShortLabel(k) }}
                  <span v-if="s.duration_ms > 0" class="mini-time">{{ (s.duration_ms / 1000).toFixed(0) }}s</span>
                </span>
              </div>
              <!-- 该候选人最近一次失败原因 (取第一个失败的 stage) -->
              <div v-if="cp.status === 'failed'" class="candidate-error">
                <span class="err-icon">⚠️</span>
                <span v-for="s in cp.stages" :key="s.error">
                  <span v-if="s.error" class="err-text">{{ stageShortLabel(Object.keys(cp.stages).find(k => cp.stages[k] === s)) }}: {{ s.error }}</span>
                </span>
              </div>
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
            <div class="tab" :class="{ active: tab === 'hitl' }" @click="toggleHitlTab()">👤 HR 复核<span v-if="summary.hitl_count > 0" class="hitl-badge">{{ summary.hitl_count }}</span></div>
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
              <details v-if="c.reflection_messages?.length" class="reflection-details">
                <summary>🤖 反思对话（{{ c.reflection_messages.length }} 条消息）</summary>
                <div class="reflection-thread">
                  <div v-for="(m, mi) in c.reflection_messages" :key="mi"
                       class="reflection-turn" :class="m.source.toLowerCase()">
                    <div class="turn-header">
                      <span class="turn-avatar">{{ turnAvatar(m.source) }}</span>
                      <span class="turn-source">{{ m.source }}</span>
                      <span class="turn-round">第 {{ m.round }} 轮</span>
                      <span class="turn-type">{{ turnTypeLabel(m.type) }}</span>
                    </div>
                    <div class="turn-content">{{ m.content }}</div>
                  </div>
                </div>
              </details>
              <details v-if="c.interview_questions?.questions?.length">
                <summary>💬 定制面试题（{{ c.interview_questions.questions.length }} 道）</summary>
                <div style="margin-top: 10px;">
                  <div class="interview-rationale">{{ c.interview_questions.rationale }}</div>
                  <div v-for="(q, qi) in c.interview_questions.questions" :key="qi" class="question-item">
                    <div class="question-head">
                      <span class="question-num">{{ qi + 1 }}</span>
                      <span class="question-cat">{{ questionCategory(q.category) }}</span>
                      <span class="question-diff" :class="q.difficulty">{{ questionDifficulty(q.difficulty) }}</span>
                      <span class="question-skill">{{ q.target_skill }}</span>
                    </div>
                    <div class="question-text">{{ q.question }}</div>
                    <div v-if="q.expected_answer_outline" class="question-answer">
                      参考答案: {{ Array.isArray(q.expected_answer_outline) ? q.expected_answer_outline.join('; ') : q.expected_answer_outline }}
                    </div>
                  </div>
                </div>
              </details>
            </div>
          </div>

          <div v-if="tab === 'hitl'">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
              <span style="font-size: 13px; color: var(--text-secondary);">待复核候选人（{{ hitlPending.length }} 人）</span>
              <button class="secondary" style="width: auto; padding: 5px 14px; font-size: 12px;" @click="loadHitlPending" :disabled="hitlLoading">
                {{ hitlLoading ? '加载中...' : '刷新' }}
              </button>
            </div>
            <div v-if="hitlPending.length === 0" class="empty" style="padding: 30px 20px;">
              暂无待复核候选人
            </div>
            <div v-for="item in hitlPending" :key="item.candidate_name" class="hitl-card">
              <div class="hitl-card-head">
                <div>
                  <strong>{{ item.candidate_name }}</strong>
                  <span style="color: var(--text-tertiary); font-size: 12px; margin-left: 8px;">{{ item.job_title }}</span>
                </div>
                <div class="hitl-original">
                  原始分 <strong>{{ item.original_score }}</strong>
                  · {{ recText(item.original_recommendation) }}
                </div>
              </div>
              <div v-if="item.hr_note" class="hitl-reason">{{ item.hr_note }}</div>
              <div class="hitl-form">
                <div class="hitl-field">
                  <label>调整分数</label>
                  <input type="text" v-model="item._adjusted_score" placeholder="0-100，留空不改" />
                </div>
                <div class="hitl-field">
                  <label>调整推荐</label>
                  <select v-model="item._recommendation">
                    <option value="">不改</option>
                    <option value="strong_recommend">强烈推荐</option>
                    <option value="recommend">推荐</option>
                    <option value="neutral">中性</option>
                    <option value="not_recommend">不推荐</option>
                  </select>
                </div>
                <div class="hitl-field" style="flex: 2;">
                  <label>备注</label>
                  <input type="text" v-model="item._note" placeholder="可选，填写审核意见" />
                </div>
                <button
                  class="secondary"
                  style="width: auto; padding: 7px 18px; font-size: 13px; white-space: nowrap; align-self: flex-end;"
                  :disabled="hitlSubmitting[item.candidate_name]"
                  @click="submitHitlDecision(item)"
                >
                  {{ hitlSubmitting[item.candidate_name] ? '提交中...' : '提交决策' }}
                </button>
              </div>
              <div v-if="hitlResult[item.candidate_name] === 'ok'" class="hitl-result ok">已提交</div>
              <div v-if="hitlResult[item.candidate_name] === 'error'" class="hitl-result err">提交失败</div>
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
