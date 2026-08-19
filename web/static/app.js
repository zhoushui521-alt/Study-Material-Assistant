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

let requestInProgress = false;
let materialRequestInProgress = false;
let stagedUpload = null;

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
    setApiState("online", "本地 API 已连接（不代表模型已调用）");
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

function materialMessageForStatus(status) {
  if (status === 429) {
    return "资料操作过于频繁、服务正忙或费用保护已触发，请稍后再试。";
  }
  if (status === 413) {
    return "文件超过 10 MiB 限制。";
  }
  if (status === 409) {
    return "同名资料状态冲突，请检查新增或替换操作。";
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
    item.textContent = `${file.name} · ${formatFileSize(file.size)}`;
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
    result.textContent = `成功：${String(item.filename)} · ${Number(item.chunk_count) || 0} 个文本块`;
    stageFileResults.append(result);
  }
  for (const item of failures) {
    const result = document.createElement("li");
    result.textContent = `失败：${String(item.filename)} · ${String(item.reason)}`;
    stageFileResults.append(result);
  }
  stageFilename.textContent = staged.length
    ? `成功暂存 ${staged.length} 个文件`
    : "本批资料全部失败";
  stageDetails.textContent = `成功 ${staged.length} 个，失败 ${failures.length} 个；失败文件未进入索引。`;
  const totalChunks = Number(payload.total_chunks) || 0;
  const totalBatches = Number(payload.embedding_batch_count) || 0;
  stageCost.textContent = staged.length
    ? `成功文件共 ${totalChunks} 个文本块，按当前配置约 ${totalBatches} 个 Embedding 批次（单次上限 60）。`
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
    for (const material of materials) {
      const item = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = `${String(material.filename)} · ${formatFileSize(material.size_bytes)}`;
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
          "部分文件解析成功，请查看逐文件结果；成功文件尚未调用 Embedding。",
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
      showMaterialMessage("error", materialMessageForStatus(response.status), requestId);
      return;
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
    stageFilename.textContent = `${payload.filename} · ${operationLabel} · ${formatFileSize(payload.size_bytes)}`;
    stageDetails.textContent = `解析为 ${payload.document_units} 个资料单元、${payload.chunk_count} 个文本块。`;
    stageFileResults.replaceChildren();
    confirmIndexButton.hidden = false;
    stageCost.textContent = `网页预览不会产生模型费用；确认后约 ${payload.embedding_batch_count} 个 Embedding 批次（单次上限 60）。`;
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
    webPreviewMeta.textContent = `Markdown ${String(payload.markdown || "").length} 字；重定向 ${redirectCount} 次；内容哈希 ${String(payload.content_sha256 || "").slice(0, 12)}…`;
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
      showMaterialMessage("error", materialMessageForStatus(response.status), requestId);
      return;
    }
    showMaterialMessage(
      "success",
      `${batchItems.length ? `${batchItems.length} 个文件` : "资料"}索引完成：新增或更新 ${payload.added}，删除旧记录 ${payload.deleted}，未变化 ${payload.unchanged}。`,
      requestId,
    );
    stagedUpload = null;
    stageSummary.hidden = true;
    webPreview.hidden = true;
    uploadForm.reset();
    renderSelectedFiles();
    webPreviewForm.reset();
    await loadMaterials();
  } catch {
    showMaterialMessage("error", "无法连接资料索引 API。");
  } finally {
    setMaterialBusy(false);
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
      showMaterialMessage("error", materialMessageForStatus(response.status), requestId);
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
  sourcesList.replaceChildren();
  requestMeta.textContent = "";
}

function renderResult(payload, requestId) {
  answer.textContent = typeof payload.answer === "string" ? payload.answer : "";
  requestMeta.textContent = requestId ? `请求 ID：${requestId}` : "";

  const sources = Array.isArray(payload.sources) ? payload.sources : [];
  sourcesList.replaceChildren();
  for (const source of sources) {
    const item = document.createElement("li");
    item.textContent = String(source);
    sourcesList.append(item);
  }
  emptySources.hidden = sources.length > 0;
  resultPanel.hidden = false;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  resultPanel.scrollIntoView({
    behavior: reducedMotion ? "auto" : "smooth",
    block: "start",
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
    let payload = {};
    try {
      payload = await response.json();
    } catch {
      payload = {};
    }

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

updateCharacterCount();
void checkHealth();
void loadMaterials();
