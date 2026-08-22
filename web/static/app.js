const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const charCount = document.querySelector("#char-count");
const submitButton = document.querySelector("#submit-button");
const buttonLabel = document.querySelector("#button-label");
const apiState = document.querySelector("#api-state");
const apiStateText = document.querySelector("#api-state-text");
const errorPanel = document.querySelector("#error-panel");
const errorMessage = document.querySelector("#error-message");
const errorRequestMeta = document.querySelector("#error-request-meta");
const resultPanel = document.querySelector("#result-panel");
const answer = document.querySelector("#answer");
const citationList = document.querySelector("#citation-list");
const sourcesList = document.querySelector("#sources-list");
const emptySources = document.querySelector("#empty-sources");
const requestMeta = document.querySelector("#request-meta");
const uploadForm = document.querySelector("#upload-form");
const materialFileInput = document.querySelector("#material-file");
const selectedFiles = document.querySelector("#selected-files");
const selectedFileCount = document.querySelector("#selected-file-count");
const selectedFileList = document.querySelector("#selected-file-list");
const uploadOperation = document.querySelector("#upload-operation");
const stageButton = document.querySelector("#stage-button");
const stageSummary = document.querySelector("#stage-summary");
const stageFilename = document.querySelector("#stage-filename");
const stageDetails = document.querySelector("#stage-details");
const stageFileResults = document.querySelector("#stage-file-results");
const stageCost = document.querySelector("#stage-cost");
const confirmIndexButton = document.querySelector("#confirm-index-button");
const materialMessage = document.querySelector("#material-message");
const materialMessageText = document.querySelector("#material-message-text");
const materialRequestMeta = document.querySelector("#material-request-meta");
const materialList = document.querySelector("#material-list");
const materialCount = document.querySelector("#material-count");
const emptyMaterials = document.querySelector("#empty-materials");

const webPreviewForm = document.querySelector("#web-preview-form");
const webMaterialUrl = document.querySelector("#web-material-url");
const webOperation = document.querySelector("#web-operation");
const webPreviewButton = document.querySelector("#web-preview-button");
const webPreview = document.querySelector("#web-preview");
const webPreviewTitle = document.querySelector("#web-preview-title");
const webPreviewLink = document.querySelector("#web-preview-link");
const webPreviewMeta = document.querySelector("#web-preview-meta");
const webPreviewMarkdown = document.querySelector("#web-preview-markdown");

const ragTab = document.querySelector("#rag-tab");
const tutorTab = document.querySelector("#tutor-tab");
const ragPanel = document.querySelector("#rag-panel");
const tutorPanel = document.querySelector("#tutor-panel");
const tutorSetup = document.querySelector("#tutor-setup");
const tutorSessionForm = document.querySelector("#tutor-session-form");
const tutorTopicInput = document.querySelector("#tutor-topic");
const tutorSessionButton = document.querySelector("#tutor-session-button");
const tutorChat = document.querySelector("#tutor-chat");
const tutorThread = document.querySelector("#tutor-thread");
const tutorEmpty = document.querySelector("#tutor-empty");
const tutorForm = document.querySelector("#tutor-form");
const tutorMessageInput = document.querySelector("#tutor-message");
const tutorCostConfirm = document.querySelector("#tutor-cost-confirm");
const tutorSubmit = document.querySelector("#tutor-submit");
const tutorMessageState = document.querySelector("#tutor-message-state");
const currentTopic = document.querySelector("#current-topic");
const learningAction = document.querySelector("#learning-action");
const learningActionNote = document.querySelector("#learning-action-note");
const learningHistoryList = document.querySelector("#learning-history-list");
const emptyLearningHistory = document.querySelector("#empty-learning-history");
const refreshHistoryButton = document.querySelector("#refresh-history-button");
const newSessionButton = document.querySelector("#new-session-button");
const focusTriggers = Array.from(document.querySelectorAll("[data-focus-target]"));
const tutorSuggestions = Array.from(document.querySelectorAll(".tutor-suggestion"));

const STORAGE_KEYS = {
  userId: "zhixing.user_id",
  sessionId: "zhixing.session_id",
  topic: "zhixing.topic",
};

const ACTION_LABELS = {
  answer_question: "回答资料问题",
  explain_concept: "解释当前概念",
  practice_quiz: "完成一次理解练习",
  summarize_learning: "整理本次学习",
  create_study_plan: "执行下一步计划",
  insufficient_evidence: "补充可用资料",
};

const INTENT_LABELS = {
  knowledge_qa: "资料问答",
  explanation: "概念解释",
  quiz: "理解练习",
  summary: "学习总结",
  study_plan: "学习规划",
};

let requestInProgress = false;
let materialRequestInProgress = false;
let tutorRequestInProgress = false;
let stagedUpload = null;
let tutorIdentity = readTutorIdentity();

function updateCharacterCount() {
  charCount.textContent = `${questionInput.value.length} / 2000`;
}

function setApiState(state, message) {
  apiState.dataset.state = state;
  apiStateText.textContent = message;
}

async function checkHealth() {
  try {
    const response = await fetch("/health", {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error("health check failed");
    }
    setApiState("online", "本地 API 已连接，不代表模型已调用");
  } catch {
    setApiState("offline", "本地 API 暂不可用");
  }
}

function messageForStatus(status) {
  if (status === 429) {
    return "请求过于频繁、服务正忙或当前进程额度已用完，请稍后再试。";
  }
  if (status === 422) {
    return "问题格式无效，请输入 1 到 2000 个字符后重试。";
  }
  if (status === 502) {
    return "资料检索或模型回答失败，请稍后重试。";
  }
  if (status === 503) {
    return "RAG 服务暂不可用，请检查本地配置和向量库。";
  }
  if (status >= 500) {
    return "服务器内部错误，请使用请求 ID 对照控制台日志。";
  }
  return "请求未成功，请检查输入后重试。";
}

function tutorMessageForStatus(status) {
  if (status === 404) {
    return "当前学习身份或 Session 已失效，请开始一个新主题。";
  }
  if (status === 429) {
    return "Tutor 请求过于频繁、服务正忙或费用保护已触发，请稍后再试。";
  }
  if (status === 422) {
    return "Tutor 输入或费用确认无效，请检查后重试。";
  }
  if (status === 502) {
    return "Tutor 处理失败，未能返回可用结果。";
  }
  if (status === 503) {
    return "本地学习数据服务暂不可用。";
  }
  if (status === 504) {
    return "Tutor 处理超时，请稍后重试。";
  }
  return "Tutor 请求没有完成，请稍后重试。";
}

function materialMessageForStatus(status, detail = "") {
  if (status === 429) {
    return "资料操作过于频繁、服务正忙或费用保护已触发，请稍后再试。";
  }
  if (status === 413) {
    return "文件超过 10 MiB 限制。";
  }
  if (status === 409) {
    return detail || "同名资料状态冲突，请检查新增或替换操作。";
  }
  if (status === 422) {
    return "文件名、类型、内容或确认参数无效。";
  }
  if (status === 502) {
    return "资料索引失败，服务已尝试恢复原有状态。";
  }
  if (status === 503) {
    return "资料与索引可能需要人工检查，请暂停继续操作。";
  }
  return "资料操作未完成，请检查输入后重试。";
}

function webMessageForStatus(status) {
  if (status === 429) {
    return "网页预览过于频繁或服务正忙，请稍后再试。";
  }
  if (status === 413) {
    return "网页响应或 Markdown 超过预览安全上限。";
  }
  if (status === 409) {
    return "该网页资料已存在，请明确选择重新抓取并替换。";
  }
  if (status === 422) {
    return "URL、网页类型、重定向目标或生成内容不符合导入要求。";
  }
  if (status === 502) {
    return "公开网页抓取失败，请检查网址或稍后重试。";
  }
  if (status === 503) {
    return "网页 Markdown 预览组件暂不可用。";
  }
  return "网页预览没有完成，请检查输入后重试。";
}

function formatFileSize(sizeBytes) {
  if (!Number.isFinite(sizeBytes) || sizeBytes < 0) {
    return "未知大小";
  }
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }
  return `${(sizeBytes / 1024).toFixed(1)} KiB`;
}

function formatDate(value) {
  const parsed = new Date(String(value || ""));
  if (Number.isNaN(parsed.getTime())) {
    return "时间未知";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(parsed);
}

function showMaterialMessage(state, message, requestId = "") {
  materialMessage.dataset.state = state;
  materialMessageText.textContent = message;
  materialRequestMeta.textContent = requestId ? `请求 ID：${requestId}` : "";
  materialRequestMeta.hidden = !requestId;
  materialMessage.hidden = false;
}

function setMaterialBusy(busy) {
  materialRequestInProgress = busy;
  materialFileInput.disabled = busy;
  uploadOperation.disabled = busy;
  stageButton.disabled = busy;
  confirmIndexButton.disabled = busy;
  webMaterialUrl.disabled = busy;
  webOperation.disabled = busy;
  webPreviewButton.disabled = busy;
}

function clearStagedMaterial({ clearWebPreview = true } = {}) {
  stagedUpload = null;
  stageSummary.hidden = true;
  stageFileResults.replaceChildren();
  confirmIndexButton.hidden = false;
  if (clearWebPreview) {
    webPreview.hidden = true;
    webPreviewTitle.textContent = "";
    webPreviewLink.textContent = "";
    webPreviewLink.removeAttribute("href");
    webPreviewMeta.textContent = "";
    webPreviewMarkdown.textContent = "";
  }
}

function renderSelectedFiles() {
  const files = Array.from(materialFileInput.files || []);
  selectedFileList.replaceChildren();
  selectedFileCount.textContent = files.length ? `已选择 ${files.length} 个文件` : "";
  for (const file of files) {
    const item = document.createElement("li");
    item.textContent = `${file.name}，${formatFileSize(file.size)}`;
    selectedFileList.append(item);
  }
  selectedFiles.hidden = files.length === 0;
}

function renderBatchStageResult(payload) {
  const staged = Array.isArray(payload.staged) ? payload.staged : [];
  const failures = Array.isArray(payload.failures) ? payload.failures : [];
  stageFileResults.replaceChildren();
  for (const item of staged) {
    const result = document.createElement("li");
    result.textContent = `成功：${String(item.filename)}，${Number(item.chunk_count) || 0} 个文本块`;
    stageFileResults.append(result);
  }
  for (const item of failures) {
    const result = document.createElement("li");
    result.textContent = `失败：${String(item.filename)}，${String(item.reason)}`;
    stageFileResults.append(result);
  }
  stageFilename.textContent = staged.length
    ? `成功暂存 ${staged.length} 个文件`
    : "本批资料全部失败";
  stageDetails.textContent = `成功 ${staged.length} 个，失败 ${failures.length} 个。失败文件未进入索引。`;
  const totalChunks = Number(payload.total_chunks) || 0;
  const totalBatches = Number(payload.embedding_batch_count) || 0;
  stageCost.textContent = staged.length
    ? `成功文件共 ${totalChunks} 个文本块，按当前配置约 ${totalBatches} 个 Embedding 批次，单次上限 60。`
    : "没有可确认写入索引的文件。";
  confirmIndexButton.hidden = staged.length === 0;
  stageSummary.hidden = false;
  return staged;
}

function safeExternalHttpUrl(value) {
  try {
    const parsed = new URL(String(value));
    return parsed.protocol === "http:" || parsed.protocol === "https:"
      ? parsed.href
      : "";
  } catch {
    return "";
  }
}

async function loadMaterials() {
  try {
    const response = await fetch("/api/materials", {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error("materials unavailable");
    }
    const payload = await response.json();
    const materials = Array.isArray(payload.materials) ? payload.materials : [];
    materialList.replaceChildren();
    materialCount.textContent = String(materials.length);
    for (const material of materials) {
      const item = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = `${String(material.filename)}，${formatFileSize(material.size_bytes)}`;
      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.textContent = "删除";
      deleteButton.setAttribute("aria-label", `删除 ${String(material.filename)}`);
      deleteButton.addEventListener("click", () => {
        void deleteMaterial(String(material.filename), deleteButton);
      });
      item.append(label, deleteButton);
      materialList.append(item);
    }
    emptyMaterials.hidden = materials.length > 0;
  } catch {
    materialCount.textContent = "0";
    showMaterialMessage("error", "无法读取当前资料列表。");
  }
}

async function stageMaterial() {
  const files = Array.from(materialFileInput.files || []);
  if (!files.length) {
    showMaterialMessage("error", "请先选择至少一个资料文件。");
    materialFileInput.focus();
    return;
  }
  if (files.length > 20) {
    showMaterialMessage("error", "单次最多选择 20 个资料文件。");
    materialFileInput.focus();
    return;
  }
  if (materialRequestInProgress) {
    return;
  }

  setMaterialBusy(true);
  materialMessage.hidden = true;
  clearStagedMaterial();
  const body = new FormData();
  for (const file of files) {
    body.append("files", file);
  }
  body.append("operation", uploadOperation.value);
  stageButton.textContent = "正在处理资料…";
  try {
    const response = await fetch("/api/materials/stage-batch", {
      method: "POST",
      headers: { Accept: "application/json" },
      body,
    });
    const requestId = response.headers.get("X-Request-ID") || "";
    const payload = await response.json().catch(() => ({}));
    if (Array.isArray(payload.staged) && Array.isArray(payload.failures)) {
      const staged = renderBatchStageResult(payload);
      stagedUpload = staged.length ? { type: "batch", staged } : null;
      if (payload.status === "partial") {
        showMaterialMessage(
          "error",
          "部分文件解析成功，请查看逐文件结果。成功文件尚未调用 Embedding。",
          requestId,
        );
      } else if (payload.status === "failed") {
        showMaterialMessage("error", "本批文件全部解析失败，请查看原因。", requestId);
      } else {
        showMaterialMessage(
          "success",
          "全部文件本地校验与解析完成，尚未调用 Embedding。",
          requestId,
        );
      }
      return;
    }
    if (!response.ok) {
      showMaterialMessage(
        "error",
        materialMessageForStatus(response.status, payload.detail),
        requestId,
      );
    }
  } catch {
    showMaterialMessage("error", "无法连接资料上传 API。");
  } finally {
    stageButton.textContent = "本地校验与解析";
    setMaterialBusy(false);
  }
}

async function previewWebMaterial() {
  const url = webMaterialUrl.value.trim();
  if (!url) {
    showMaterialMessage("error", "请先输入公开网页 URL。");
    webMaterialUrl.focus();
    return;
  }
  if (materialRequestInProgress) {
    return;
  }

  setMaterialBusy(true);
  materialMessage.hidden = true;
  clearStagedMaterial();
  try {
    const response = await fetch("/api/web-materials/preview", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url, operation: webOperation.value }),
    });
    const requestId = response.headers.get("X-Request-ID") || "";
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      showMaterialMessage("error", webMessageForStatus(response.status), requestId);
      return;
    }

    stagedUpload = payload;
    const operationLabel = payload.operation === "replace" ? "替换网页资料" : "新增网页资料";
    stageFilename.textContent = `${payload.filename}，${operationLabel}，${formatFileSize(payload.size_bytes)}`;
    stageDetails.textContent = `解析为 ${payload.document_units} 个资料单元、${payload.chunk_count} 个文本块。`;
    stageFileResults.replaceChildren();
    confirmIndexButton.hidden = false;
    stageCost.textContent = `网页预览不会产生模型费用。确认后约 ${payload.embedding_batch_count} 个 Embedding 批次，单次上限 60。`;
    stageSummary.hidden = false;

    webPreviewTitle.textContent = String(payload.title || "网页资料");
    const safeUrl = safeExternalHttpUrl(payload.canonical_url);
    webPreviewLink.textContent = safeUrl || "来源 URL 无法展示";
    if (safeUrl) {
      webPreviewLink.href = safeUrl;
    } else {
      webPreviewLink.removeAttribute("href");
    }
    const redirectCount = Number.isInteger(payload.redirect_count)
      ? payload.redirect_count
      : 0;
    webPreviewMeta.textContent = `Markdown ${String(payload.markdown || "").length} 字，重定向 ${redirectCount} 次，内容哈希 ${String(payload.content_sha256 || "").slice(0, 12)}…`;
    webPreviewMarkdown.textContent = String(payload.markdown || "");
    webPreview.hidden = false;
    showMaterialMessage(
      "success",
      "网页安全抓取与 Markdown 预览完成，尚未调用 Embedding。",
      requestId,
    );
  } catch {
    showMaterialMessage("error", "无法连接网页预览 API。");
  } finally {
    setMaterialBusy(false);
  }
}

async function confirmIndex() {
  const batchItems = Array.isArray(stagedUpload?.staged)
    ? stagedUpload.staged
    : [];
  if ((!stagedUpload?.upload_id && !batchItems.length) || materialRequestInProgress) {
    return;
  }
  setMaterialBusy(true);
  try {
    const isBatch = batchItems.length > 0;
    const target = isBatch
      ? "/api/materials/batch/index"
      : `/api/materials/${encodeURIComponent(stagedUpload.upload_id)}/index`;
    const requestBody = isBatch
      ? {
          upload_ids: batchItems.map((item) => String(item.upload_id)),
          confirm_api_cost: true,
        }
      : { confirm_api_cost: true };
    const response = await fetch(target, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });
    const requestId = response.headers.get("X-Request-ID") || "";
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      showMaterialMessage(
        "error",
        materialMessageForStatus(response.status, payload.detail),
        requestId,
      );
      return;
    }
    const jobId = String(payload.job_id || "");
    if (!jobId) {
      showMaterialMessage("error", "任务创建成功，但响应中缺少任务 ID。", requestId);
      return;
    }
    showMaterialMessage(
      "success",
      `${batchItems.length ? `${batchItems.length} 个文件` : "资料"}已进入后台任务，任务编号 ${jobId.slice(0, 8)}…`,
      requestId,
    );
    const completedJob = await waitForDocumentJob(jobId);
    if (completedJob.status === "failed") {
      showMaterialMessage(
        "error",
        String(completedJob.error_message || "资料后台处理失败。"),
        requestId,
      );
      return;
    }
    const result = completedJob.result || {};
    showMaterialMessage(
      "success",
      `${batchItems.length ? `${batchItems.length} 个文件` : "资料"}索引完成：新增或更新 ${Number(result.added || 0)}，删除旧记录 ${Number(result.deleted || 0)}，未变化 ${Number(result.unchanged || 0)}。`,
      requestId,
    );
    stagedUpload = null;
    stageSummary.hidden = true;
    webPreview.hidden = true;
    uploadForm.reset();
    renderSelectedFiles();
    webPreviewForm.reset();
    await loadMaterials();
  } catch (error) {
    const message = error instanceof Error && error.message
      ? error.message
      : "无法连接资料索引 API。";
    showMaterialMessage("error", message);
  } finally {
    setMaterialBusy(false);
  }
}

async function waitForDocumentJob(jobId) {
  while (true) {
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
      headers: { Accept: "application/json" },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(String(payload.detail || "任务状态查询失败。"));
    }
    if (payload.status === "completed" || payload.status === "failed") {
      return payload;
    }
    showMaterialMessage(
      "success",
      `后台正在处理资料：${Number(payload.progress || 0)}%，任务 ${jobId.slice(0, 8)}…`,
    );
  }
}

async function deleteMaterial(filename, button) {
  if (materialRequestInProgress) {
    return;
  }
  const confirmed = window.confirm(
    `确认删除“${filename}”及其索引记录？该操作不会调用 Embedding。`,
  );
  if (!confirmed) {
    return;
  }
  setMaterialBusy(true);
  button.disabled = true;
  try {
    const response = await fetch(`/api/materials/${encodeURIComponent(filename)}`, {
      method: "DELETE",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ confirm_delete: true }),
    });
    const requestId = response.headers.get("X-Request-ID") || "";
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      showMaterialMessage(
        "error",
        materialMessageForStatus(response.status, payload.detail),
        requestId,
      );
      return;
    }
    showMaterialMessage(
      "success",
      `已删除 ${payload.filename} 和 ${payload.deleted_records} 条索引记录。`,
      requestId,
    );
    await loadMaterials();
  } catch {
    showMaterialMessage("error", "无法连接资料删除 API。");
  } finally {
    setMaterialBusy(false);
    button.disabled = false;
  }
}

function showError(message, requestId = "") {
  errorMessage.textContent = message;
  errorRequestMeta.textContent = requestId ? `请求 ID：${requestId}` : "";
  errorRequestMeta.hidden = !requestId;
  errorPanel.hidden = false;
}

function clearOutput() {
  errorPanel.hidden = true;
  resultPanel.hidden = true;
  answer.textContent = "";
  citationList.replaceChildren();
  sourcesList.replaceChildren();
  requestMeta.textContent = "";
}

function createCitationCard(citation) {
  const card = document.createElement("article");
  card.className = "citation-card";

  const header = document.createElement("div");
  header.className = "citation-card-header";
  const filename = document.createElement("strong");
  filename.textContent = String(citation.filename || citation.source || "未命名资料");
  const location = document.createElement("span");
  const page = Number.isInteger(citation.page) ? `第 ${citation.page} 页` : "无固定页码";
  location.textContent = `${String(citation.citation_id || "Citation")}，${page}`;
  header.append(filename, location);

  const excerpt = document.createElement("p");
  excerpt.className = "citation-excerpt";
  excerpt.textContent = String(citation.excerpt || "当前 Citation 没有返回摘录。");

  const locator = document.createElement("p");
  locator.className = "citation-locator";
  locator.textContent = `定位：${String(citation.locator || "未提供")}`;

  card.append(header, excerpt, locator);
  return card;
}

function renderCitations(citations, sources, target = citationList) {
  target.replaceChildren();
  sourcesList.replaceChildren();
  const normalizedCitations = Array.isArray(citations) ? citations : [];
  const normalizedSources = Array.isArray(sources) ? sources : [];

  for (const citation of normalizedCitations) {
    target.append(createCitationCard(citation));
  }

  if (!normalizedCitations.length) {
    for (const source of normalizedSources) {
      const item = document.createElement("li");
      item.textContent = String(source);
      sourcesList.append(item);
    }
  }

  sourcesList.hidden = normalizedCitations.length > 0 || normalizedSources.length === 0;
  emptySources.hidden = normalizedCitations.length > 0 || normalizedSources.length > 0;
}

function renderResult(payload, requestId) {
  answer.textContent = typeof payload.answer === "string" ? payload.answer : "";
  requestMeta.textContent = requestId ? `请求 ID：${requestId}` : "";
  renderCitations(payload.citations, payload.sources);
  resultPanel.hidden = false;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  resultPanel.scrollIntoView({
    behavior: reducedMotion ? "auto" : "smooth",
    block: "nearest",
  });
}

async function submitQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    showError("问题不能为空。请先输入你想从资料中查找的内容。");
    questionInput.focus();
    return;
  }
  if (requestInProgress) {
    return;
  }

  requestInProgress = true;
  clearOutput();
  submitButton.disabled = true;
  submitButton.setAttribute("aria-busy", "true");
  buttonLabel.textContent = "正在检索与回答…";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });
    const requestId = response.headers.get("X-Request-ID") || "";
    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      showError(messageForStatus(response.status), requestId);
      return;
    }
    renderResult(payload, requestId);
  } catch {
    showError("无法连接本地 API，请确认 Uvicorn 仍在运行。");
    setApiState("offline", "本地 API 暂不可用");
  } finally {
    requestInProgress = false;
    submitButton.disabled = false;
    submitButton.removeAttribute("aria-busy");
    buttonLabel.textContent = "向资料提问";
  }
}

function activateMode(mode) {
  const showTutor = mode === "tutor";
  ragTab.setAttribute("aria-selected", String(!showTutor));
  tutorTab.setAttribute("aria-selected", String(showTutor));
  ragPanel.hidden = showTutor;
  tutorPanel.hidden = !showTutor;
  if (showTutor) {
    if (tutorIdentity.sessionId) {
      tutorMessageInput.focus();
    } else {
      tutorTopicInput.focus();
    }
  } else {
    questionInput.focus();
  }
}

function safeStorageGet(key) {
  try {
    return window.localStorage.getItem(key) || "";
  } catch {
    return "";
  }
}

function safeStorageSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // 本地存储不可用时仍允许当前页面继续使用 Session。
  }
}

function safeStorageRemove(key) {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // 本地存储不可用时只清理当前内存状态。
  }
}

function readTutorIdentity() {
  return {
    userId: safeStorageGet(STORAGE_KEYS.userId),
    sessionId: safeStorageGet(STORAGE_KEYS.sessionId),
    topic: safeStorageGet(STORAGE_KEYS.topic),
  };
}

function persistTutorIdentity() {
  if (tutorIdentity.userId) {
    safeStorageSet(STORAGE_KEYS.userId, tutorIdentity.userId);
  }
  if (tutorIdentity.sessionId) {
    safeStorageSet(STORAGE_KEYS.sessionId, tutorIdentity.sessionId);
  }
  if (tutorIdentity.topic) {
    safeStorageSet(STORAGE_KEYS.topic, tutorIdentity.topic);
  }
}

function clearTutorSession() {
  tutorIdentity.sessionId = "";
  tutorIdentity.topic = "";
  safeStorageRemove(STORAGE_KEYS.sessionId);
  safeStorageRemove(STORAGE_KEYS.topic);
  tutorChat.hidden = true;
  tutorSetup.hidden = false;
  currentTopic.textContent = "还没有学习主题";
  learningAction.textContent = "等待第一次 Tutor 互动";
  learningActionNote.textContent = "Tutor 返回的真实学习行动会显示在这里。";
  newSessionButton.hidden = true;
  learningHistoryList.replaceChildren();
  emptyLearningHistory.hidden = false;
  tutorMessageState.hidden = true;
  tutorTopicInput.value = "";
}

function setTutorSessionReady() {
  tutorSetup.hidden = true;
  tutorChat.hidden = false;
  currentTopic.textContent = tutorIdentity.topic || "未命名主题";
  newSessionButton.hidden = false;
}

async function ensureTutorUser() {
  if (tutorIdentity.userId) {
    return tutorIdentity.userId;
  }
  const response = await fetch("/api/users", {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || !payload.user_id) {
    throw new Error("无法创建本地学习身份。");
  }
  tutorIdentity.userId = String(payload.user_id);
  safeStorageSet(STORAGE_KEYS.userId, tutorIdentity.userId);
  return tutorIdentity.userId;
}

async function createTutorSession(topic) {
  if (tutorRequestInProgress) {
    return;
  }
  tutorRequestInProgress = true;
  tutorSessionButton.disabled = true;
  tutorSessionButton.textContent = "正在建立学习空间…";
  tutorMessageState.hidden = true;
  try {
    const userId = await ensureTutorUser();
    const response = await fetch(`/api/users/${encodeURIComponent(userId)}/sessions`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ topic }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.session_id) {
      throw new Error("无法创建学习 Session。");
    }
    tutorIdentity = {
      userId,
      sessionId: String(payload.session_id),
      topic: String(payload.topic || topic),
    };
    persistTutorIdentity();
    setTutorSessionReady();
    clearTutorThread();
    renderLearningHistory([]);
    tutorMessageInput.focus();
  } catch (error) {
    showTutorState(
      error instanceof Error ? error.message : "无法建立学习空间。",
      "error",
    );
  } finally {
    tutorRequestInProgress = false;
    tutorSessionButton.disabled = false;
    tutorSessionButton.textContent = "开始这个主题";
  }
}

function clearTutorThread() {
  for (const message of tutorThread.querySelectorAll(".thread-message")) {
    message.remove();
  }
  tutorEmpty.hidden = false;
}

function appendThreadMessage(role, content, intent = "") {
  tutorEmpty.hidden = true;
  const wrapper = document.createElement("article");
  wrapper.className = "thread-message";
  wrapper.dataset.role = role;

  const label = document.createElement("span");
  label.textContent = role === "user"
    ? "你"
    : `AI Tutor${intent ? `，${INTENT_LABELS[intent] || intent}` : ""}`;
  const text = document.createElement("p");
  text.textContent = String(content || "");
  wrapper.append(label, text);
  tutorThread.append(wrapper);
  tutorThread.scrollTop = tutorThread.scrollHeight;
  return wrapper;
}

function appendTutorExtras(container, payload) {
  const citations = Array.isArray(payload.citations) ? payload.citations : [];
  const quiz = payload.quiz && typeof payload.quiz === "object" ? payload.quiz : null;
  const summary = payload.summary && typeof payload.summary === "object" ? payload.summary : null;

  if (citations.length) {
    const extras = document.createElement("div");
    extras.className = "thread-extras";
    const heading = document.createElement("strong");
    heading.textContent = "可核对 Citation";
    extras.append(heading);
    for (const citation of citations) {
      extras.append(createCitationCard(citation));
    }
    container.append(extras);
  }

  if (quiz) {
    const extras = document.createElement("div");
    extras.className = "thread-extras";
    const heading = document.createElement("strong");
    heading.textContent = "理解练习";
    const question = document.createElement("p");
    question.textContent = String(quiz.question || "");
    extras.append(heading, question);
    const options = Array.isArray(quiz.options) ? quiz.options : [];
    if (options.length) {
      const list = document.createElement("ul");
      for (const option of options) {
        const item = document.createElement("li");
        item.textContent = String(option);
        list.append(item);
      }
      extras.append(list);
    }
    container.append(extras);
  }

  if (summary) {
    const extras = document.createElement("div");
    extras.className = "thread-extras";
    const heading = document.createElement("strong");
    heading.textContent = "学习总结";
    const text = document.createElement("p");
    text.textContent = String(summary.summary || "");
    extras.append(heading, text);
    const nextSteps = Array.isArray(summary.next_steps) ? summary.next_steps : [];
    if (nextSteps.length) {
      const list = document.createElement("ul");
      for (const step of nextSteps) {
        const item = document.createElement("li");
        item.textContent = String(step);
        list.append(item);
      }
      extras.append(list);
    }
    container.append(extras);
  }
}

function showTutorState(message, state = "error") {
  tutorMessageState.textContent = message;
  tutorMessageState.dataset.state = state;
  tutorMessageState.hidden = false;
}

function updateLearningAction(payload) {
  const action = String(payload.learning_action || "");
  learningAction.textContent = ACTION_LABELS[action] || action || "等待下一次 Tutor 互动";
  const intent = INTENT_LABELS[String(payload.intent || "")] || "学习";
  learningActionNote.textContent = `本次由 Tutor 的“${intent}”路径返回。`;
  if (payload.topic) {
    tutorIdentity.topic = String(payload.topic);
    currentTopic.textContent = tutorIdentity.topic;
    safeStorageSet(STORAGE_KEYS.topic, tutorIdentity.topic);
  }
}

function renderLearningHistory(records) {
  const normalized = Array.isArray(records) ? records : [];
  const recent = [...normalized]
    .sort((left, right) => new Date(right.created_at) - new Date(left.created_at))
    .slice(0, 6);
  learningHistoryList.replaceChildren();
  for (const record of recent) {
    const item = document.createElement("li");
    const action = ACTION_LABELS[String(record.activity_type || "")]
      || INTENT_LABELS[String(record.activity_type || "")]
      || String(record.activity_type || "学习活动");
    item.append(document.createTextNode(`${String(record.topic || "未命名主题")}：${action}`));
    const time = document.createElement("time");
    time.dateTime = String(record.created_at || "");
    time.textContent = formatDate(record.created_at);
    item.append(time);
    learningHistoryList.append(item);
  }
  emptyLearningHistory.hidden = recent.length > 0;
}

function renderTutorHistory(messages) {
  clearTutorThread();
  const normalized = Array.isArray(messages) ? messages : [];
  const recent = [...normalized]
    .sort((left, right) => new Date(left.created_at) - new Date(right.created_at))
    .slice(-20);
  for (const message of recent) {
    appendThreadMessage(
      String(message.role || "") === "tutor" ? "tutor" : "user",
      String(message.content || ""),
      String(message.intent || ""),
    );
  }
}

async function loadLearningHistory({ renderThread = true } = {}) {
  if (!tutorIdentity.userId || !tutorIdentity.sessionId) {
    renderLearningHistory([]);
    return;
  }
  refreshHistoryButton.disabled = true;
  try {
    const target = `/api/users/${encodeURIComponent(tutorIdentity.userId)}/history?session_id=${encodeURIComponent(tutorIdentity.sessionId)}`;
    const response = await fetch(target, {
      headers: { Accept: "application/json" },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 404) {
        clearTutorSession();
      }
      throw new Error("无法读取最近学习记录。");
    }
    if (renderThread) {
      renderTutorHistory(payload.messages);
    }
    renderLearningHistory(payload.learning_records);
  } catch (error) {
    showTutorState(
      error instanceof Error ? error.message : "无法读取最近学习记录。",
      "error",
    );
  } finally {
    refreshHistoryButton.disabled = false;
  }
}

async function submitTutorMessage() {
  const message = tutorMessageInput.value.trim();
  if (!tutorIdentity.userId || !tutorIdentity.sessionId) {
    clearTutorSession();
    showTutorState("请先创建一个学习主题。", "error");
    tutorTopicInput.focus();
    return;
  }
  if (!message) {
    showTutorState("请输入你想和 Tutor 学习的内容。", "error");
    tutorMessageInput.focus();
    return;
  }
  if (!tutorCostConfirm.checked) {
    showTutorState("请先确认本次 Tutor 请求会产生模型 API 费用。", "error");
    tutorCostConfirm.focus();
    return;
  }
  if (tutorRequestInProgress) {
    return;
  }

  tutorRequestInProgress = true;
  tutorSubmit.disabled = true;
  tutorSubmit.textContent = "Tutor 正在处理…";
  tutorMessageState.hidden = true;

  try {
    const response = await fetch("/api/tutor/chat", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: tutorIdentity.userId,
        session_id: tutorIdentity.sessionId,
        message,
        confirm_api_cost: true,
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 404) {
        clearTutorSession();
      }
      showTutorState(tutorMessageForStatus(response.status), "error");
      return;
    }

    appendThreadMessage("user", message);
    const tutorReply = appendThreadMessage(
      "tutor",
      String(payload.answer || ""),
      String(payload.intent || ""),
    );
    appendTutorExtras(tutorReply, payload);
    updateLearningAction(payload);
    tutorMessageInput.value = "";
    tutorCostConfirm.checked = false;
    showTutorState("Tutor 已完成本次学习请求，结果已写入本地学习记录。", "success");
    await loadLearningHistory({ renderThread: false });
  } catch {
    showTutorState("无法连接 Tutor API，请确认本地服务仍在运行。", "error");
  } finally {
    tutorRequestInProgress = false;
    tutorSubmit.disabled = false;
    tutorSubmit.textContent = "发送给 AI Tutor";
  }
}

function initializeTutor() {
  if (tutorIdentity.sessionId && tutorIdentity.userId) {
    setTutorSessionReady();
    void loadLearningHistory();
  } else {
    clearTutorSession();
  }
}

function initializeRevealAnimations() {
  const targets = Array.from(document.querySelectorAll("[data-reveal]"));
  if (!targets.length) {
    return;
  }
  document.documentElement.classList.add("reveal-ready");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || !("IntersectionObserver" in window)) {
    for (const target of targets) {
      target.classList.add("is-visible");
    }
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      }
    },
    {
      rootMargin: "0px 0px -10% 0px",
      threshold: 0.12,
    },
  );
  for (const target of targets) {
    observer.observe(target);
  }
}

function focusLearningTarget(trigger) {
  const targetSelector = trigger.dataset.focusTarget;
  const target = targetSelector ? document.querySelector(targetSelector) : null;
  if (!target) {
    return;
  }
  activateMode("rag");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  target.scrollIntoView({
    behavior: reduceMotion ? "auto" : "smooth",
    block: "center",
  });
  window.setTimeout(() => target.focus(), reduceMotion ? 0 : 420);
}

function openTutorSuggestion(trigger) {
  const prompt = String(trigger.dataset.tutorPrompt || "").trim();
  const topic = String(trigger.dataset.topic || "").trim();
  activateMode("tutor");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.querySelector("#learn").scrollIntoView({
    behavior: reduceMotion ? "auto" : "smooth",
    block: "start",
  });

  if (tutorIdentity.sessionId && tutorIdentity.userId) {
    tutorMessageInput.value = prompt;
    window.setTimeout(() => tutorMessageInput.focus(), reduceMotion ? 0 : 420);
    return;
  }

  if (topic) {
    tutorTopicInput.value = topic;
  }
  window.setTimeout(() => tutorTopicInput.focus(), reduceMotion ? 0 : 420);
}

questionInput.addEventListener("input", updateCharacterCount);
questionInput.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") {
    event.preventDefault();
    form.requestSubmit();
  }
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  void submitQuestion();
});
uploadForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void stageMaterial();
});
materialFileInput.addEventListener("change", () => {
  clearStagedMaterial();
  renderSelectedFiles();
});
uploadOperation.addEventListener("change", () => {
  clearStagedMaterial();
});
webPreviewForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void previewWebMaterial();
});
webMaterialUrl.addEventListener("input", () => {
  clearStagedMaterial();
});
webOperation.addEventListener("change", () => {
  clearStagedMaterial();
});
confirmIndexButton.addEventListener("click", () => {
  void confirmIndex();
});

ragTab.addEventListener("click", () => {
  activateMode("rag");
});
tutorTab.addEventListener("click", () => {
  activateMode("tutor");
});
tutorSessionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const topic = tutorTopicInput.value.trim();
  if (!topic) {
    tutorTopicInput.focus();
    return;
  }
  void createTutorSession(topic);
});
tutorForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void submitTutorMessage();
});
newSessionButton.addEventListener("click", () => {
  clearTutorSession();
  activateMode("tutor");
});
refreshHistoryButton.addEventListener("click", () => {
  void loadLearningHistory();
});
for (const trigger of focusTriggers) {
  trigger.addEventListener("click", (event) => {
    event.preventDefault();
    focusLearningTarget(trigger);
  });
}
for (const trigger of tutorSuggestions) {
  trigger.addEventListener("click", () => {
    openTutorSuggestion(trigger);
  });
}

initializeRevealAnimations();
updateCharacterCount();
initializeTutor();
void checkHealth();
void loadMaterials();
