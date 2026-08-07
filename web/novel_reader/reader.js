const $ = (id) => document.getElementById(id);

const state = {
  books: [],
  book: null,
  serviceSettings: null,
  serviceSettingsName: "正在读取服务设置",
  serviceSettingsSource: "default",
  serviceSettingsFingerprint: "",
  authRedirecting: false,
  bookRefreshSerial: 0,
  bookRefreshApplied: 0,
  streamRunId: 0,
  qualityContext: null,
  stopRequested: false,
  chapterIndex: 0,
  selectedBlock: null,
  playingBlock: null,
  listeningMode: "stream",
  playbackMode: "",
  playbackEpoch: null,
  playbackControlStopping: false,
  qualityQueue: [],
  qualityJobs: new Set(),
  qualityPlaybackReject: null,
  qualityAudio: new Audio(),
  audioContext: null,
  streamAbort: null,
  currentJob: "",
  stopped: false,
  paused: false,
  pendingImport: null,
  editingBookId: "",
  pendingBookDeleteId: "",
  bookManagerTab: "all",
  pollTimer: null,
  editMode: false,
};

function api(path) {
  return path.startsWith("/") ? path : `/${path}`;
}

async function jsonFetch(path, options = {}) {
  const response = await fetch(api(path), options);
  if (response.status === 401 && !state.authRedirecting) {
    state.authRedirecting = true;
    const next = `${window.location.pathname}${window.location.search}`;
    window.location.assign(`/login?next=${encodeURIComponent(next)}`);
    throw new DOMException("Authentication required", "AbortError");
  }
  if (!response.ok) {
    let message = await response.text();
    try { message = JSON.parse(message).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function toast(message, error = false) {
  const item = document.createElement("div");
  item.className = `toast${error ? " error" : ""}`;
  item.textContent = message;
  $("toast-region").appendChild(item);
  window.setTimeout(() => item.remove(), 3600);
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds || 0)));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return minutes >= 60
    ? `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `${minutes}:${String(rest).padStart(2, "0")}`;
}

function mediaUrl(bookId, path) {
  return `/api/document-projects/${encodeURIComponent(bookId)}/media?path=${encodeURIComponent(path)}`;
}

function currentChapter() {
  return state.book?.chapters?.[state.chapterIndex] || null;
}

function chapterSegments(chapter = currentChapter()) {
  if (!state.book || !chapter) return [];
  return state.book.segments.filter(
    (segment) => segment.index >= chapter.segment_start && segment.index <= chapter.segment_end
  );
}

async function loadHealth() {
  try {
    const health = await jsonFetch("/api/health");
    const pill = $("service-pill");
    pill.classList.toggle("ready", health.state === "ready");
    pill.lastChild.textContent = health.state === "ready"
      ? `${health.active_profile_label || "TTS Metal"} · STT ${health.stt?.ready ? "就绪" : "未就绪"}`
      : `服务状态：${health.state}`;
  } catch (error) {
    $("service-pill").lastChild.textContent = "本地服务未连接";
  }
}

function toBoolean(value, fallback = false) {
  if (value === undefined || value === null || value === "") return fallback;
  if (typeof value === "boolean") return value;
  return !["0", "false", "no", "off"].includes(String(value).toLowerCase());
}

function normalizeServiceSettings(shared = {}) {
  return {
    model_profile: String(shared.model_profile || "qwen_0_6b"),
    reference_audio_path: String(shared.reference_audio_path || ""),
    voice_name: String(shared.voice_name || "服务默认音色"),
    temperature: Number(shared.qwen_temperature ?? 0.9),
    top_p: Number(shared.qwen_top_p ?? 1),
    top_k: Number(shared.qwen_top_k ?? 50),
    repetition_penalty: Number(shared.qwen_repetition_penalty ?? 1.05),
    max_new_tokens: Number(shared.qwen_max_new_tokens ?? 2048),
    codec_chunk_frames: Number(shared.qwen_chunk_size ?? 8),
    seed: Number(shared.qwen_seed ?? 1234),
    qwen_clone_mode: String(shared.qwen_clone_mode || "xvec"),
    qwen_reference_text: String(shared.qwen_reference_text || ""),
    qwen_non_streaming_mode: false,
    qwen_append_silence: toBoolean(shared.qwen_append_silence, true),
    qwen_min_new_tokens: Number(shared.qwen_min_new_tokens ?? 2),
    aac_bitrate: String(shared.qwen_aac_bitrate || "80k"),
  };
}

async function loadServiceSettings() {
  const payload = await jsonFetch("/api/service-settings");
  const nextSettings = normalizeServiceSettings(payload.settings || {});
  const nextFingerprint = JSON.stringify(nextSettings);
  const changed = Boolean(
    state.serviceSettingsFingerprint
    && state.serviceSettingsFingerprint !== nextFingerprint
  );
  state.serviceSettings = nextSettings;
  state.serviceSettingsName = payload.name || nextSettings.voice_name;
  state.serviceSettingsSource = payload.source || "default";
  state.serviceSettingsFingerprint = nextFingerprint;
  updateListeningSummary();
  updateBitrateLabel();
  if (changed) {
    resetRealtimeAfterSettingsChange();
    toast(`服务音色已更新为“${state.serviceSettingsName}”`);
  }
}

async function loadBooks(preferredId = "") {
  const data = await jsonFetch("/api/document-projects");
  state.books = data.projects || [];
  renderBooks();
  const remembered = preferredId || state.book?.id || localStorage.getItem("qwen-reader-book") || "";
  const target = state.books.find((book) => book.id === remembered) || state.books[0];
  if (target) await openBook(target.id, false);
  else showEmpty();
}

function renderBooks() {
  const list = $("book-list");
  list.innerHTML = "";
  if (!state.books.length) {
    list.innerHTML = '<p class="empty-copy">书架还是空的。点击右上角“＋”导入第一本小说。</p>';
    renderBookManager();
    return;
  }
  for (const book of state.books) {
    const button = document.createElement("button");
    button.className = `book-card${state.book?.id === book.id ? " active" : ""}`;
    button.type = "button";
    const progress = Math.round(100 * Number(book.stats?.progress || 0));
    const initial = (book.name || "书").trim().slice(0, 1);
    button.innerHTML = `
      <span class="book-cover">${escapeHTML(initial)}</span>
      <span class="book-copy">
        <strong>${escapeHTML(book.name || "未命名小说")}</strong>
        <span>${book.chapters?.length || 1} 章 · ${book.stats?.total_chars || 0} 字 · ${progress}% 音频</span>
        <span class="book-progress"><i style="width:${progress}%"></i></span>
      </span>`;
    button.onclick = () => openBook(book.id);
    list.appendChild(button);
  }
  renderBookManager();
}

function bookStatusLabel(book) {
  if (book.state === "running") return "正在生成";
  if (book.state === "stopping") return "正在停止";
  if (book.state === "completed") return "音频已完成";
  if (book.state === "paused") return "已暂停";
  return "可阅读";
}

function renderBookManager() {
  const list = $("managed-book-list");
  if (!list) return;
  const completedCount = state.books.filter((book) => book.state === "completed" || book.final_audio).length;
  const displayedBooks = state.bookManagerTab === "completed"
    ? state.books.filter((book) => book.state === "completed" || book.final_audio)
    : state.books;
  $("book-manager-tab-all").textContent = `全部书籍 · ${state.books.length}`;
  $("book-manager-tab-completed").textContent = `音频已完成 · ${completedCount}`;
  $("book-manager-tab-all").classList.toggle("active", state.bookManagerTab === "all");
  $("book-manager-tab-completed").classList.toggle("active", state.bookManagerTab === "completed");
  $("book-manager-tab-all").setAttribute("aria-selected", String(state.bookManagerTab === "all"));
  $("book-manager-tab-completed").setAttribute("aria-selected", String(state.bookManagerTab === "completed"));
  $("library-manager-summary").textContent = state.books.length
    ? "管理书名、阅读状态和已经制作的整书音频"
    : "还没有导入任何书";
  list.innerHTML = "";
  if (!displayedBooks.length) {
    list.innerHTML = `
      <div class="manager-empty-state">
        <span>${state.bookManagerTab === "completed" ? "✓" : "书"}</span>
        <strong>${state.bookManagerTab === "completed" ? "还没有制作完成的整书音频" : "书架还是空的"}</strong>
        <p>${state.bookManagerTab === "completed" ? "完成整本书生成后，会集中显示在这里。" : "导入 TXT、Markdown 或 DOCX，系统会自动拆分章节。"}</p>
      </div>`;
    return;
  }
  for (const book of displayedBooks) {
    const progress = Math.round(100 * Number(book.stats?.progress || 0));
    const active = ["running", "stopping"].includes(book.state);
    const editing = state.editingBookId === book.id;
    const confirmingDelete = state.pendingBookDeleteId === book.id;
    const row = document.createElement("article");
    row.className = `managed-book-row${state.book?.id === book.id ? " current" : ""}${confirmingDelete ? " confirming-delete" : ""}`;
    row.innerHTML = `
      <span class="manager-book-cover" aria-hidden="true">
        <span>${escapeHTML((book.name || "书").trim().slice(0, 1))}</span>
      </span>
      <div class="manager-book-copy">
        ${editing ? `
          <input class="manager-rename-input" type="text" maxlength="120" value="${escapeHTML(book.name || "")}" aria-label="新书名">
        ` : `
          <span class="manager-book-title">
            <strong>${escapeHTML(book.name || "未命名小说")}</strong>
            ${state.book?.id === book.id ? '<span class="manager-kind-badge">当前阅读</span>' : ""}
            ${book.final_audio ? '<span class="manager-kind-badge completed">整书完成</span>' : ""}
          </span>
        `}
        <span class="manager-book-meta">
          <span>${book.chapters?.length || 1} 章</span>
          <span>${book.stats?.total_chars || 0} 字</span>
          <span>音频 ${progress}%</span>
          <span class="manager-book-state${active ? " active" : ""}">${bookStatusLabel(book)}</span>
        </span>
      </div>
      <div class="manager-book-actions"></div>`;
    const actions = row.querySelector(".manager-book-actions");
    if (editing) {
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "manager-action-button";
      cancel.textContent = "取消";
      cancel.onclick = () => {
        state.editingBookId = "";
        renderBookManager();
      };
      const save = document.createElement("button");
      save.type = "button";
      save.className = "manager-action-button primary";
      save.textContent = "保存";
      save.onclick = () => renameManagedBook(book.id, row.querySelector(".manager-rename-input").value);
      actions.append(cancel, save);
      window.setTimeout(() => {
        const input = row.querySelector(".manager-rename-input");
        input?.focus();
        input?.select();
        input?.addEventListener("keydown", (event) => {
          if (event.key === "Enter") save.click();
          if (event.key === "Escape") cancel.click();
        });
      }, 0);
    } else if (confirmingDelete) {
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "manager-action-button";
      cancel.textContent = "取消";
      cancel.onclick = () => {
        state.pendingBookDeleteId = "";
        renderBookManager();
      };
      const confirm = document.createElement("button");
      confirm.type = "button";
      confirm.className = "manager-action-button danger";
      confirm.textContent = "确认删除";
      confirm.onclick = () => deleteManagedBook(book.id);
      actions.append(cancel, confirm);
    } else {
      const open = document.createElement("button");
      open.type = "button";
      open.className = "manager-action-button primary";
      open.textContent = state.book?.id === book.id ? "当前" : "打开";
      open.disabled = state.book?.id === book.id;
      open.onclick = async () => {
        await openBook(book.id);
        $("library-manager-dialog").close();
      };
      const rename = document.createElement("button");
      rename.type = "button";
      rename.className = "manager-action-button";
      rename.textContent = "重命名";
      rename.onclick = () => {
        state.editingBookId = book.id;
        state.pendingBookDeleteId = "";
        renderBookManager();
      };
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "manager-action-button danger-quiet";
      remove.textContent = "删除";
      remove.disabled = active;
      remove.title = active ? "请先停止生成，再删除这本书" : "删除书籍及其生成音频";
      remove.onclick = () => {
        state.pendingBookDeleteId = book.id;
        state.editingBookId = "";
        renderBookManager();
      };
      actions.append(open, rename, remove);
    }
    list.appendChild(row);
  }
}

async function renameManagedBook(bookId, name) {
  const revised = String(name || "").replace(/\s+/g, " ").trim();
  if (!revised) return toast("书名不能为空", true);
  try {
    const updated = await jsonFetch(`/api/document-projects/${encodeURIComponent(bookId)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: revised }),
    });
    state.books = state.books.map((book) => book.id === bookId ? updated : book);
    if (state.book?.id === bookId) state.book = updated;
    state.editingBookId = "";
    renderBooks();
    if (state.book?.id === bookId) renderBook();
    toast(`书名已更新为《${updated.name}》`);
  } catch (error) {
    toast(error.message, true);
  }
}

async function deleteManagedBook(bookId) {
  const book = state.books.find((item) => item.id === bookId);
  try {
    if (state.book?.id === bookId) await stopPlayback(false);
    await jsonFetch(`/api/document-projects/${encodeURIComponent(bookId)}`, { method: "DELETE" });
    if (state.book?.id === bookId) {
      state.book = null;
      localStorage.removeItem("qwen-reader-book");
    }
    state.pendingBookDeleteId = "";
    toast(`《${book?.name || "这本书"}》已从本机删除`);
    await loadBooks();
  } catch (error) {
    toast(error.message, true);
  }
}

async function openBook(bookId, remember = true) {
  await stopPlayback(false);
  state.book = await jsonFetch(`/api/document-projects/${encodeURIComponent(bookId)}`);
  syncControlsFromBook();
  state.chapterIndex = Math.min(
    Math.max(0, Number(localStorage.getItem(`qwen-reader-chapter-${bookId}`) || 0)),
    Math.max(0, (state.book.chapters?.length || 1) - 1)
  );
  state.selectedBlock = Number(state.book.playback?.segment_index ?? currentChapter()?.segment_start ?? 0);
  if (remember) localStorage.setItem("qwen-reader-book", bookId);
  renderBooks();
  renderBook();
}

function syncControlsFromBook() {
  updateListeningSummary();
  updateBitrateLabel();
}

function showEmpty() {
  state.book = null;
  $("empty-state").classList.remove("hidden");
  $("reading-view").classList.add("hidden");
  $("book-title").textContent = "我的书架";
  $("book-kicker").textContent = "选择一本小说开始阅读";
  $("chapter-list").innerHTML = '<p class="empty-copy">打开小说后显示章节。</p>';
  setActionsEnabled(false);
}

function renderBook() {
  if (!state.book) return showEmpty();
  $("empty-state").classList.add("hidden");
  $("reading-view").classList.remove("hidden");
  $("book-title").textContent = state.book.name;
  $("book-kicker").textContent = `${state.serviceSettingsName} · ${state.book.chapters?.length || 1} 章`;
  const download = $("download-whole-book");
  if (state.book.final_audio) {
    download.href = mediaUrl(state.book.id, state.book.final_audio);
    download.download = `${state.book.name}.m4a`;
    download.classList.remove("hidden");
  } else {
    download.removeAttribute("href");
    download.classList.add("hidden");
  }
  renderChapters();
  renderCurrentChapter();
  renderWholeBookProgress();
  setActionsEnabled(true);
}

function renderChapters() {
  const container = $("chapter-list");
  container.innerHTML = "";
  for (const chapter of state.book.chapters || []) {
    const segments = state.book.segments.filter(
      (segment) => segment.index >= chapter.segment_start && segment.index <= chapter.segment_end
    );
    const completed = segments.filter((segment) => segment.status === "completed").length;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chapter-button${chapter.index === state.chapterIndex ? " active" : ""}`;
    button.innerHTML = `
      <span class="chapter-number">${String(chapter.index + 1).padStart(2, "0")}</span>
      <strong>${escapeHTML(chapter.title)}</strong>
      <small>${segments.length} 块 · ${completed}/${segments.length} AAC</small>`;
    button.onclick = () => selectChapter(chapter.index);
    container.appendChild(button);
  }
}

function renderCurrentChapter() {
  const chapter = currentChapter();
  if (!chapter) return;
  const segments = chapterSegments(chapter);
  $("chapter-position").textContent = `CHAPTER ${String(chapter.index + 1).padStart(2, "0")}`;
  $("chapter-title").textContent = chapter.title;
  $("chapter-progress").textContent = `${segments.length} 个文字块`;
  const duration = segments.reduce((sum, segment) => sum + Number(segment.duration_seconds || 0), 0);
  $("chapter-duration").textContent = duration ? `已生成 ${formatDuration(duration)} AAC` : "尚未生成音频";
  $("previous-chapter").disabled = chapter.index <= 0;
  $("next-chapter").disabled = chapter.index >= state.book.chapters.length - 1;
  $("selected-range").textContent = chapter.title;
  $("range-detail").textContent = `第 ${chapter.segment_start + 1}–${chapter.segment_end + 1} 块 · 共 ${segments.length} 块`;
  const blocks = $("reader-blocks");
  blocks.innerHTML = "";
  for (const segment of segments) {
    const block = document.createElement("p");
    block.className = "reader-block";
    block.dataset.segmentIndex = segment.index;
    block.dataset.originalText = segment.text;
    block.tabIndex = 0;
    block.contentEditable = state.editMode ? "true" : "false";
    block.spellcheck = state.editMode;
    if (segment.index === state.selectedBlock) block.classList.add("selected");
    if (segment.index === state.playingBlock) block.classList.add("playing");
    if (segment.status === "generating" || segment.status === "encoding") block.classList.add("generating");
    const stateLabel = segment.status === "completed"
      ? `AAC ${formatDuration(segment.duration_seconds)}`
      : segment.status === "generating" ? "正在生成"
      : segment.status === "encoding" ? "正在转 AAC"
      : segment.status === "failed" ? "生成失败"
      : "";
    block.appendChild(document.createTextNode(segment.text));
    if (stateLabel) {
      const status = document.createElement("span");
      status.className = "block-state";
      status.contentEditable = "false";
      status.textContent = stateLabel;
      block.appendChild(status);
    }
    block.onclick = () => {
      state.selectedBlock = segment.index;
      if (!state.editMode) selectBlock(segment.index);
      else renderBlockSelection(segment.index);
    };
    block.onblur = () => {
      if (state.editMode) saveSegmentEdit(block, segment.index);
    };
    block.onkeydown = (event) => {
      if (state.editMode) {
        if (event.key === "Escape") {
          event.preventDefault();
          block.firstChild.nodeValue = block.dataset.originalText;
          block.blur();
        } else if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
          event.preventDefault();
          block.blur();
        }
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectBlock(segment.index);
      }
    };
    blocks.appendChild(block);
  }
}

function renderBlockSelection(index) {
  document.querySelectorAll(".reader-block").forEach((block) => {
    block.classList.toggle("selected", Number(block.dataset.segmentIndex) === Number(index));
  });
}

function editableBlockText(block) {
  const clone = block.cloneNode(true);
  clone.querySelectorAll(".block-state").forEach((item) => item.remove());
  return clone.innerText.replace(/\s+/g, " ").trim();
}

async function saveSegmentEdit(block, segmentIndex) {
  const text = editableBlockText(block);
  const original = block.dataset.originalText || "";
  if (!text || text === original) {
    if (!text) block.firstChild.nodeValue = original;
    return;
  }
  block.contentEditable = "false";
  block.classList.add("generating");
  try {
    state.book = await jsonFetch(
      `/api/document-projects/${encodeURIComponent(state.book.id)}/segments/${segmentIndex}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }
    );
    toast(`第 ${segmentIndex + 1} 块已保存；旧 AAC 已失效`);
    renderBook();
  } catch (error) {
    toast(error.message, true);
    block.firstChild.nodeValue = original;
    block.contentEditable = "true";
    block.classList.remove("generating");
  }
}

async function toggleEditMode() {
  if (!state.book) return;
  if (!state.editMode) await stopPlayback();
  state.editMode = !state.editMode;
  document.body.classList.toggle("edit-mode", state.editMode);
  const button = $("edit-mode-button");
  button.classList.toggle("active", state.editMode);
  button.setAttribute("aria-pressed", String(state.editMode));
  button.textContent = state.editMode ? "✓ 完成编辑" : "✎ 编辑正文";
  renderCurrentChapter();
  toast(state.editMode ? "编辑模式已开启；离开文字块时自动保存" : "已回到浏览模式");
}

async function selectChapter(index) {
  if (!state.book) return;
  const audioActive = Boolean(
    state.currentJob
    || state.qualityJobs.size
    || ["stream", "quality", "quality-online", "preview"].includes(state.playbackMode)
  );
  if (audioActive) await stopPlayback(true);
  state.chapterIndex = Math.max(0, Math.min(index, state.book.chapters.length - 1));
  state.selectedBlock = currentChapter().segment_start;
  localStorage.setItem(`qwen-reader-chapter-${state.book.id}`, state.chapterIndex);
  renderChapters();
  renderCurrentChapter();
  document.body.classList.remove("toc-open");
  $("toc-toggle").classList.remove("active");
  $("toc-toggle").setAttribute("aria-expanded", "false");
  document.querySelector(".reader-main")?.scrollTo({ top: 0, behavior: "smooth" });
}

function selectBlock(index) {
  state.selectedBlock = index;
  renderCurrentChapter();
  if (
    ["running", "stopping"].includes(state.book?.state)
    || state.playbackMode === "quality-generating"
  ) {
    return;
  }
  const segment = state.book?.segments?.find((item) => item.index === index);
  if (segment?.status === "completed" && segment.audio_file) {
    playQualitySegments([segment]);
  }
}

function setActionsEnabled(enabled) {
  const generating = Boolean(
    state.currentJob
    || state.qualityJobs.size
    || ["running", "stopping"].includes(state.book?.state)
    || state.playbackMode === "quality-generating"
    || state.playbackMode === "quality-online"
  );
  $("stream-listen").disabled = !enabled || generating;
  $("quality-generate").disabled = !enabled || generating;
  $("start-listening").disabled = !enabled || generating;
  $("generate-whole-book").disabled = !enabled || generating;
  $("stop-generation").disabled = !generating;
  $("previous-block").disabled = !enabled;
  $("next-block").disabled = !enabled;
}

function setListeningMode(mode, persist = true) {
  state.listeningMode = mode === "quality" ? "quality" : "stream";
  const qualitySelected = state.listeningMode === "quality";
  $("stream-listen").classList.toggle("selected", !qualitySelected);
  $("stream-listen").setAttribute("aria-pressed", String(!qualitySelected));
  $("quality-generate").classList.toggle("selected", qualitySelected);
  $("quality-generate").setAttribute("aria-pressed", String(qualitySelected));
  $("selected-listening-mode").textContent = qualitySelected
    ? `当前：高质量整块 · AAC ${selectedBookBitrate()}`
    : "当前：实时流式";
  if (!state.playbackMode) {
    $("playback-mode").textContent = qualitySelected ? "待播 · 高质量整块" : "待播 · 实时流式";
  }
  if (persist) localStorage.setItem("qwen-reader-listening-mode", state.listeningMode);
}

async function startSelectedListening() {
  $("listening-controls-dialog").close();
  if (state.listeningMode === "quality") {
    await startQualityChapter();
  } else {
    await startStreamChapter();
  }
}

function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function openImport(file) {
  if (!file) return;
  state.pendingImport = file;
  $("import-file-name").textContent = file.name;
  $("import-file-detail").textContent = `${(file.size / 1024).toFixed(1)} KB · 将自动识别章节`;
  $("import-name").value = file.name.replace(/\.[^.]+$/, "");
  $("import-dialog").showModal();
}

function settingsSnapshot() {
  return {
    ...(state.serviceSettings || normalizeServiceSettings({})),
  };
}

function hexToRGBA(hex, alpha) {
  const clean = String(hex).replace("#", "");
  const value = Number.parseInt(clean, 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

function applyAppearance() {
  const theme = $("theme-select").value;
  const accent = $("accent-color").value;
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.setProperty("--accent", accent);
  document.documentElement.style.setProperty("--accent-strong", accent);
  document.documentElement.style.setProperty("--accent-soft", hexToRGBA(accent, .11));
  $("accent-value").textContent = accent.toUpperCase();
  localStorage.setItem("qwen-reader-theme", theme);
  localStorage.setItem("qwen-reader-accent", accent);
}

function updateBitrateLabel() {
  const bitrate = $("book-aac-bitrate")?.value || state.serviceSettings?.aac_bitrate || "80k";
  $("quality-bitrate-label").textContent = `逐段非流式 · 双路预取 · AAC ${bitrate}`;
  updateListeningSummary();
}

function updateListeningSummary() {
  const settings = state.serviceSettings || normalizeServiceSettings({});
  const model = settings.model_profile === "qwen_1_7b" ? "1.7B 高质量" : "0.6B 极速";
  const seed = Number(settings.seed);
  if ($("active-preset-name")) {
    $("active-preset-name").textContent = state.serviceSettingsSource === "studio" ? "音频工作台设置" : state.serviceSettingsName;
  }
  if ($("active-voice-name")) $("active-voice-name").textContent = settings.voice_name;
  if ($("active-generation-summary")) {
    $("active-generation-summary").textContent = `${model} · ${seed < 0 ? "随机 Seed" : `Seed ${seed}`} · AAC ${settings.aac_bitrate}`;
  }
  if ($("service-setting-name")) $("service-setting-name").textContent = state.serviceSettingsName;
  if ($("service-setting-detail")) {
    $("service-setting-detail").textContent = `${settings.voice_name} · ${model} · ${settings.qwen_clone_mode === "icl" ? "ICL" : "X-vector"}`;
  }
  if ($("import-service-setting")) {
    $("import-service-setting").textContent = `${settings.voice_name} · ${model}`;
  }
}

async function resetRealtimeAfterSettingsChange() {
  const hadRealtimeAudio = Boolean(
    state.currentJob
    || state.playbackMode === "stream"
    || state.playbackMode === "preview"
    || (state.playbackMode === "quality" && state.qualityAudio.src)
  );
  if (hadRealtimeAudio) {
    await stopPlayback(true);
    state.playbackMode = "";
    state.playingBlock = null;
    $("play-toggle").textContent = "▶";
    $("playback-mode").textContent = "参数已更新";
    $("playback-seed").textContent = "可重新实时试听";
    renderCurrentChapter();
  }
  setActionsEnabled(Boolean(state.book));
}

function qualitySettingsChanged(settings) {
  const saved = state.book?.settings;
  if (!saved) return true;
  const currentConfiguredSeed = Number(settings.seed);
  const savedConfiguredSeed = Number(saved.configured_seed ?? saved.seed);
  return (
    settings.model_profile !== saved.model_profile
    || settings.reference_audio_path !== saved.reference_audio_path
    || settings.qwen_clone_mode !== saved.qwen_clone_mode
    || settings.qwen_reference_text.trim() !== String(saved.qwen_reference_text || "").trim()
    || currentConfiguredSeed !== savedConfiguredSeed
    || settings.aac_bitrate !== (saved.aac_bitrate || "80k")
    || Number(settings.temperature) !== Number(saved.temperature ?? 0.9)
    || Number(settings.top_p) !== Number(saved.top_p ?? 1)
    || Number(settings.top_k) !== Number(saved.top_k ?? 50)
    || Number(settings.repetition_penalty) !== Number(saved.repetition_penalty ?? 1.05)
    || Number(settings.max_new_tokens) !== Number(saved.max_new_tokens ?? 2048)
    || Number(settings.codec_chunk_frames) !== Number(saved.codec_chunk_frames ?? 8)
    || Number(settings.qwen_min_new_tokens) !== Number(saved.qwen_min_new_tokens ?? 2)
    || Boolean(settings.qwen_append_silence) !== toBoolean(saved.qwen_append_silence, true)
  );
}

async function confirmImport() {
  const file = state.pendingImport;
  if (!file) return;
  const button = $("confirm-import");
  button.disabled = true;
  button.textContent = "正在解析章节…";
  try {
    const settings = settingsSnapshot();
    const form = new FormData();
    form.append("document", file, file.name);
    form.append("name", $("import-name").value.trim());
    form.append("max_chars", $("import-block-size").value);
    form.append("settings_json", JSON.stringify(settings));
    const book = await jsonFetch("/api/document-projects", { method: "POST", body: form });
    $("import-dialog").close();
    state.pendingImport = null;
    toast(`《${book.name}》已加入书架，识别到 ${book.chapters?.length || 1} 章`);
    await loadBooks(book.id);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "解析并加入书架";
  }
}

async function refreshCurrentBook(render = true) {
  if (!state.book) return null;
  const bookId = state.book.id;
  const requestSerial = ++state.bookRefreshSerial;
  const refreshed = await jsonFetch(`/api/document-projects/${encodeURIComponent(bookId)}`);
  if (
    state.book?.id !== bookId
    || requestSerial < state.bookRefreshApplied
  ) {
    return state.book;
  }
  state.bookRefreshApplied = requestSerial;
  state.book = refreshed;
  if (render) renderBook();
  return state.book;
}

function selectedBookBitrate() {
  return $("book-aac-bitrate")?.value || "80k";
}

async function beginQualityGeneration() {
  if (!state.book || !state.book.segments?.length) return;
  await stopPlayback(false);
  $("stream-listen").disabled = true;
  $("quality-generate").disabled = true;
  $("generate-whole-book").disabled = true;
  const segmentStart = 0;
  const segmentEnd = state.book.segments.length - 1;
  const form = new FormData();
  form.append("segment_start", segmentStart);
  form.append("segment_end", segmentEnd);
  const settings = settingsSnapshot();
  settings.qwen_non_streaming_mode = true;
  settings.aac_bitrate = selectedBookBitrate();
  if (qualitySettingsChanged(settings)) {
    form.append("settings_json", JSON.stringify(settings));
  }
  state.qualityContext = {
    scope: "book",
    title: state.book.name,
    segmentStart,
    segmentEnd,
    autoPlay: false,
  };
  state.stopRequested = false;
  try {
    state.book = await jsonFetch(
      `/api/document-projects/${encodeURIComponent(state.book.id)}/start-selection`,
      { method: "POST", body: form }
    );
    state.playbackMode = "quality-generating";
    renderBook();
    setActionsEnabled(true);
    $("generation-label").textContent = "正在生成整本书";
    pollQualityGeneration();
  } catch (error) {
    state.qualityContext = null;
    state.stopRequested = false;
    state.playbackMode = "";
    setActionsEnabled(true);
    toast(error.message, true);
  }
}

function generateQualityChapter() {
  return startQualityChapter();
}

function generateWholeBook() {
  return beginQualityGeneration();
}

function waitForJob(jobId, runId) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + 10 * 60 * 1000;
    const check = async () => {
      try {
        if (Date.now() > deadline) {
          reject(new Error("该文字块生成超过 10 分钟，任务已停止"));
          return;
        }
        const status = await jsonFetch(`/api/generate-stream/${encodeURIComponent(jobId)}/status`);
        if (state.stopped || runId !== state.streamRunId || !state.qualityJobs.has(jobId)) {
          reject(new DOMException("Stopped", "AbortError"));
        } else if (status.state === "finished" && status.result_ready) {
          resolve(status);
        } else if (["error", "closed", "interrupted"].includes(status.state)) {
          reject(new Error(status.error || "该文字块生成失败"));
        } else {
          window.setTimeout(check, 350);
        }
      } catch (error) {
        reject(error);
      }
    };
    check();
  });
}

function playTemporaryAAC(prepared) {
  return new Promise((resolve, reject) => {
    const audio = state.qualityAudio;
    const fail = (error) => {
      state.qualityPlaybackReject = null;
      reject(error);
    };
    state.qualityPlaybackReject = fail;
    audio.src = prepared.objectURL;
    audio.playbackRate = Number($("playback-rate").value || 1);
    audio.onended = () => {
      state.qualityPlaybackReject = null;
      resolve();
    };
    audio.onerror = () => fail(new Error("临时 AAC 无法播放"));
    audio.play().then(() => {
      $("play-toggle").textContent = "Ⅱ";
    }).catch(fail);
  });
}

async function removeTemporaryAudio(jobId) {
  if (!jobId) return;
  await fetch(`/api/generate-stream/${encodeURIComponent(jobId)}/ephemeral-audio`, {
    method: "DELETE",
  }).catch(() => {});
  state.qualityJobs.delete(jobId);
  if (state.currentJob === jobId) {
    state.currentJob = [...state.qualityJobs][0] || "";
  }
}

async function prepareQualityBlock(segment, runId) {
  const form = streamForm(segment);
  form.set("streaming_generation", "0");
  form.set("qwen_non_streaming_mode", "1");
  form.set("ephemeral_audio", "1");
  const started = await jsonFetch("/api/generate-stream/start", { method: "POST", body: form });
  if (Number.isFinite(Number(started.playback_epoch))) {
    state.playbackEpoch = Number(started.playback_epoch);
  }
  const jobId = started.job_id;
  state.qualityJobs.add(jobId);
  state.currentJob = jobId;
  setActionsEnabled(true);
  try {
    await waitForJob(jobId, runId);
    if (state.stopped || runId !== state.streamRunId) {
      throw new DOMException("Stopped", "AbortError");
    }
    const response = await fetch(
      `/api/generate-stream/${encodeURIComponent(jobId)}/result-audio-aac?bitrate=${encodeURIComponent(selectedBookBitrate())}&playback=1`
    );
    if (!response.ok) throw new Error(await response.text());
    const playbackEpoch = Number(response.headers.get("X-Playback-Epoch"));
    if (Number.isFinite(playbackEpoch)) state.playbackEpoch = playbackEpoch;
    const blob = await response.blob();
    return {
      segment,
      jobId,
      seed: started.seed,
      seedMode: started.seed_mode,
      objectURL: URL.createObjectURL(blob),
    };
  } catch (error) {
    await removeTemporaryAudio(jobId);
    throw error;
  }
}

function queueQualityBlock(segment, runId) {
  return prepareQualityBlock(segment, runId).then(
    (prepared) => ({ prepared, error: null }),
    (error) => ({ prepared: null, error })
  );
}

async function startQualityChapter() {
  const chapter = currentChapter();
  if (!state.book || !chapter) return;
  const segments = chapterSegments(chapter).filter(
    (segment) => segment.index >= (state.selectedBlock ?? chapter.segment_start)
  );
  if (!segments.length) return;
  await stopPlayback(false);
  const runId = ++state.streamRunId;
  state.stopped = false;
  state.stopRequested = false;
  state.playbackMode = "quality-online";
  setActionsEnabled(true);
  state.playingBlock = segments[0].index;
  state.selectedBlock = segments[0].index;
  updatePlayer(segments[0], "高质量非流式 · 正在完整生成", null);
  renderCurrentChapter();
  scrollPlayingBlock();
  let pendingPreparation = queueQualityBlock(segments[0], runId);
  try {
    for (let index = 0; index < segments.length; index += 1) {
      const outcome = await pendingPreparation;
      pendingPreparation = null;
      if (outcome.error) throw outcome.error;
      const prepared = outcome.prepared;
      if (state.stopped || runId !== state.streamRunId) {
        URL.revokeObjectURL(prepared.objectURL);
        await removeTemporaryAudio(prepared.jobId);
        break;
      }
      const segment = prepared.segment;
      state.playingBlock = segment.index;
      state.selectedBlock = segment.index;
      updatePlayer(segment, `高质量 AAC ${selectedBookBitrate()} · 双路预取`, prepared.seed);
      renderCurrentChapter();
      scrollPlayingBlock();
      $("playback-seed").textContent = `Seed ${prepared.seed}${prepared.seedMode === "random" ? " · 本次随机" : ""}`;
      if (index + 1 < segments.length) {
        // While the current complete AAC block is playing, use the free
        // generation lane to prepare exactly one following block.
        pendingPreparation = queueQualityBlock(segments[index + 1], runId);
      }
      try {
        await playTemporaryAAC(prepared);
        await savePlayback(segment.index, 0);
      } finally {
        state.qualityAudio.pause();
        state.qualityAudio.removeAttribute("src");
        state.qualityAudio.load();
        URL.revokeObjectURL(prepared.objectURL);
        await removeTemporaryAudio(prepared.jobId);
      }
    }
  } catch (error) {
    if (!state.stopped && error.name !== "AbortError") toast(error.message, true);
  } finally {
    if (pendingPreparation) {
      const outcome = await pendingPreparation;
      if (outcome.prepared) {
        URL.revokeObjectURL(outcome.prepared.objectURL);
        await removeTemporaryAudio(outcome.prepared.jobId);
      }
    }
    if (runId !== state.streamRunId) return;
    state.currentJob = "";
    state.playingBlock = null;
    state.playbackMode = "";
    $("play-toggle").textContent = "▶";
    $("playback-mode").textContent = state.stopped ? "生成已停止" : "本章播放完成";
    renderCurrentChapter();
    setActionsEnabled(Boolean(state.book));
  }
}

function renderWholeBookProgress() {
  if (!state.book) return;
  const segments = state.book.segments || [];
  const counts = { pending: 0, generating: 0, encoding: 0, completed: 0, failed: 0 };
  for (const segment of segments) counts[segment.status] = (counts[segment.status] || 0) + 1;
  const total = segments.length;
  const completed = counts.completed || 0;
  const progress = Number(state.book.stats?.progress ?? (total ? completed / total : 0));
  const hasProduction = completed > 0 || ["running", "stopping", "paused", "error"].includes(state.book.state);
  $("generation-progress").classList.toggle("hidden", !hasProduction);
  $("generation-bar").value = progress;
  $("generation-percent").textContent = `${Math.round(progress * 100)}%`;
  $("generation-label").textContent = state.book.message || (state.book.final_audio ? "整书音频已完成" : "整书制作进度");
  $("book-progress-detail-button").textContent = `查看详情 · ${completed}/${total} 段`;
  const eta = state.book.stats?.eta_seconds;
  $("book-progress-detail").innerHTML = `
    <span>已完成：${completed} 段</span>
    <span>总计：${total} 段</span>
    <span>生成中：${(counts.generating || 0) + (counts.encoding || 0)} 段</span>
    <span>待处理：${counts.pending || 0} 段</span>
    <span>失败：${counts.failed || 0} 段</span>
    <span>AAC：${escapeHTML(state.book.settings?.aac_bitrate || selectedBookBitrate())}</span>
    <span>已生成：${formatDuration(state.book.stats?.completed_audio_seconds || 0)}</span>
    <span>剩余：${eta == null ? "计算中" : formatDuration(eta)}</span>`;
}

async function pollQualityGeneration() {
  window.clearTimeout(state.pollTimer);
  const context = state.qualityContext;
  if (!context || !state.book) return;
  try {
    await refreshCurrentBook(true);
    const segments = state.book.segments.filter(
      (segment) => segment.index >= context.segmentStart && segment.index <= context.segmentEnd
    );
    const completed = segments.filter((segment) => segment.status === "completed").length;
    renderWholeBookProgress();
    setActionsEnabled(true);
    if (state.book.state === "error") throw new Error(state.book.message || "生成失败");
    const stillRunning = state.book.state === "running" || state.book.state === "stopping";
    if (stillRunning) {
      if (
        state.book.state === "stopping"
        && context.stopDeadline
        && Date.now() > context.stopDeadline
      ) {
        state.qualityContext = null;
        state.stopRequested = false;
        state.playbackMode = "";
        setActionsEnabled(true);
        toast("服务端停止时间较长，已转为后台监控；已完成段落不会丢失", true);
        return;
      }
      state.pollTimer = window.setTimeout(pollQualityGeneration, 1200);
      return;
    }
    state.playbackMode = "";
    setActionsEnabled(true);
    const playable = segments.filter((segment) => segment.status === "completed" && segment.audio_file);
    if (state.stopRequested || state.book.state === "paused" && completed < segments.length) {
      toast(`已停止生成；已保存 ${completed}/${segments.length} 个完整 AAC 文字块`);
    } else if (state.book.final_audio) {
      toast(`《${context.title}》整本书已生成：${playable.length} 个 AAC 文字块，整书 M4A 已合并`);
    } else {
      toast(`《${context.title}》段落已完成，但整书文件尚未就绪`, true);
    }
    state.qualityContext = null;
    state.stopRequested = false;
  } catch (error) {
    state.qualityContext = null;
    state.stopRequested = false;
    state.playbackMode = "";
    setActionsEnabled(Boolean(state.book));
    toast(error.message, true);
  }
}

function playQualitySegments(segments) {
  if (!state.book || !segments.length) return;
  stopPlayback(false).then(() => {
    state.playbackMode = "quality";
    state.qualityQueue = [...segments];
    $("playback-mode").textContent = "高质量 AAC";
    playNextQuality();
  });
}

function playNextQuality() {
  const segment = state.qualityQueue.shift();
  if (!segment || !state.book) {
    state.playingBlock = null;
    $("play-toggle").textContent = "▶";
    renderCurrentChapter();
    return;
  }
  state.playingBlock = segment.index;
  state.selectedBlock = segment.index;
  state.qualityAudio.src = mediaUrl(state.book.id, segment.audio_file);
  state.qualityAudio.playbackRate = Number($("playback-rate").value || 1);
  state.qualityAudio.onended = playNextQuality;
  state.qualityAudio.play().catch((error) => toast(error.message, true));
  updatePlayer(segment, "高质量 AAC", segment.seed);
  $("play-toggle").textContent = "Ⅱ";
  renderCurrentChapter();
  savePlayback(segment.index, 0);
  scrollPlayingBlock();
}

function streamForm(segment, seedOverride = null) {
  const settings = settingsSnapshot();
  const form = new FormData();
  form.append("mode", "voice_clone");
  form.append("language", "Chinese");
  form.append("text", segment.text);
  form.append(
    "max_new_tokens",
    String(Math.max(settings.qwen_min_new_tokens, Math.min(settings.max_new_tokens, Math.round(segment.text.length * 3.2))))
  );
  form.append("codec_chunk_frames", String(settings.codec_chunk_frames));
  form.append("seed", String(seedOverride == null ? settings.seed : seedOverride));
  form.append("temperature", String(settings.temperature));
  form.append("top_p", String(settings.top_p));
  form.append("top_k", String(settings.top_k));
  form.append("repetition_penalty", String(settings.repetition_penalty));
  form.append("model_profile", settings.model_profile);
  form.append("voice_name", settings.voice_name);
  form.append("qwen_clone_mode", settings.qwen_clone_mode);
  form.append("qwen_reference_text", settings.qwen_reference_text);
  form.append("qwen_non_streaming_mode", "0");
  form.append("qwen_append_silence", settings.qwen_append_silence ? "1" : "0");
  form.append("qwen_min_new_tokens", String(settings.qwen_min_new_tokens));
  form.append("streaming_generation", "1");
  form.append("example_audio_path", settings.reference_audio_path);
  form.append("use_service_settings", "1");
  return form;
}

async function startStreamChapter() {
  const chapter = currentChapter();
  if (!state.book || !chapter) return;
  const segments = chapterSegments(chapter).filter((segment) => segment.index >= (state.selectedBlock ?? chapter.segment_start));
  if (!segments.length) return;
  await stopPlayback(false);
  const runId = ++state.streamRunId;
  state.playbackMode = "stream";
  state.stopped = false;
  state.paused = false;
  $("stream-listen").disabled = true;
  $("playback-mode").textContent = "实时流式";
  try {
    let streamSeed = null;
    for (const segment of segments) {
      if (state.stopped || runId !== state.streamRunId) break;
      streamSeed = await streamOneBlock(segment, streamSeed);
      await savePlayback(segment.index, 0);
    }
  } catch (error) {
    if (!state.stopped && error.name !== "AbortError") toast(error.message, true);
  } finally {
    if (runId !== state.streamRunId) return;
    $("stream-listen").disabled = false;
    state.currentJob = "";
    state.playingBlock = null;
    $("play-toggle").textContent = "▶";
    renderCurrentChapter();
  }
}

async function streamOneBlock(segment, seedOverride = null) {
  state.playingBlock = segment.index;
  state.selectedBlock = segment.index;
  updatePlayer(segment, "实时流式", null);
  renderCurrentChapter();
  scrollPlayingBlock();
  const start = await jsonFetch(
    "/api/generate-stream/start",
    { method: "POST", body: streamForm(segment, seedOverride) }
  );
  if (Number.isFinite(Number(start.playback_epoch))) {
    state.playbackEpoch = Number(start.playback_epoch);
  }
  try {
    await playStreamingAudio(start);
  } finally {
    await fetch(`/api/generate-stream/${encodeURIComponent(start.job_id)}/close`, {
      method: "POST",
    }).catch(() => {});
  }
  return start.seed;
}

async function playStreamingAudio(start) {
  state.currentJob = start.job_id;
  setActionsEnabled(Boolean(state.book));
  $("playback-seed").textContent = `Seed ${start.seed}${start.seed_mode === "random" ? " · 本次随机" : ""}`;
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!state.audioContext || state.audioContext.state === "closed") {
    state.audioContext = new AudioContextCtor({ sampleRate: start.sample_rate || 24000 });
  }
  const audioContext = state.audioContext;
  await audioContext.resume();
  $("play-toggle").textContent = "Ⅱ";
  state.streamAbort = new AbortController();
  const response = await fetch(`/api/generate-stream/${start.job_id}/audio`, { signal: state.streamAbort.signal });
  if (!response.ok || !response.body) throw new Error(await response.text());
  const playbackEpoch = Number(response.headers.get("X-Playback-Epoch"));
  if (Number.isFinite(playbackEpoch)) state.playbackEpoch = playbackEpoch;
  const channels = Number(response.headers.get("X-Audio-Channels") || start.channels || 1);
  const sampleRate = Number(response.headers.get("X-Audio-Sample-Rate") || start.sample_rate || 24000);
  const reader = response.body.getReader();
  let nextTime = audioContext.currentTime + 0.08;
  let remainder = new Uint8Array(0);
  while (true) {
    let inactivityTimer;
    const read = reader.read();
    const timeout = new Promise((_, reject) => {
      inactivityTimer = window.setTimeout(() => {
        state.streamAbort?.abort();
        reject(new Error("音频流超过 60 秒没有数据，已停止任务"));
      }, 60_000);
    });
    const { value, done } = await Promise.race([read, timeout]).finally(() => {
      window.clearTimeout(inactivityTimer);
    });
    if (done || state.stopped) break;
    let bytes = value;
    if (remainder.length) {
      const joined = new Uint8Array(remainder.length + value.length);
      joined.set(remainder);
      joined.set(value, remainder.length);
      bytes = joined;
      remainder = new Uint8Array(0);
    }
    const frameBytes = 2 * channels;
    const usable = bytes.length - (bytes.length % frameBytes);
    if (usable < bytes.length) remainder = bytes.slice(usable);
    if (!usable) continue;
    const view = new DataView(bytes.buffer, bytes.byteOffset, usable);
    const frames = usable / frameBytes;
    while (
      !state.stopped
      && audioContext.state !== "closed"
      && nextTime - audioContext.currentTime > 8
    ) {
      await new Promise((resolve) => window.setTimeout(resolve, 80));
    }
    if (state.stopped || audioContext.state === "closed") break;
    const buffer = audioContext.createBuffer(channels, frames, sampleRate);
    for (let channel = 0; channel < channels; channel += 1) {
      const output = buffer.getChannelData(channel);
      for (let frame = 0; frame < frames; frame += 1) {
        output[frame] = view.getInt16((frame * channels + channel) * 2, true) / 32768;
      }
    }
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    const playbackRate = Number($("playback-rate").value || 1);
    source.playbackRate.value = playbackRate;
    source.connect(audioContext.destination);
    nextTime = Math.max(nextTime, audioContext.currentTime + 0.03);
    source.start(nextTime);
    nextTime += buffer.duration / playbackRate;
  }
  while (
    !state.stopped
    && audioContext.state !== "closed"
    && audioContext.currentTime < nextTime - 0.04
  ) {
    await new Promise((resolve) => window.setTimeout(resolve, 80));
  }
  const finalStatus = await jsonFetch(
    `/api/generate-stream/${encodeURIComponent(start.job_id)}/status`
  ).catch(() => null);
  if (finalStatus?.state === "closed") {
    if (state.audioContext && state.audioContext.state !== "closed") {
      await state.audioContext.close().catch(() => {});
    }
    state.audioContext = null;
    throw new DOMException("Stopped", "AbortError");
  }
}

async function pollPlaybackControl() {
  const audioActive = Boolean(
    state.currentJob
    || state.qualityJobs.size
    || state.qualityAudio.src
    || ["stream", "quality", "quality-online", "preview"].includes(state.playbackMode)
  );
  if (!audioActive || state.playbackControlStopping) return;
  try {
    const status = await jsonFetch("/api/playback/status");
    const epoch = Number(status.playback_epoch);
    if (!Number.isFinite(epoch)) return;
    if (state.playbackEpoch == null) {
      state.playbackEpoch = epoch;
      return;
    }
    if (epoch !== state.playbackEpoch) {
      state.playbackEpoch = epoch;
      state.playbackControlStopping = true;
      await stopPlayback(false);
      toast("后台已强制停止全部音频与运算");
    }
  } catch (_) {
    // Service restarts are handled by the existing health polling.
  } finally {
    state.playbackControlStopping = false;
  }
}

async function stopPlayback(markStopped = true) {
  state.streamRunId += 1;
  if (markStopped) state.stopped = true;
  window.clearTimeout(state.pollTimer);
  state.qualityQueue = [];
  if (state.qualityPlaybackReject) {
    state.qualityPlaybackReject(new DOMException("Stopped", "AbortError"));
    state.qualityPlaybackReject = null;
  }
  state.qualityAudio.pause();
  state.qualityAudio.removeAttribute("src");
  state.qualityAudio.load();
  if (state.streamAbort) state.streamAbort.abort();
  state.streamAbort = null;
  if (state.qualityJobs.size) {
    const jobIds = [...state.qualityJobs];
    await Promise.all(jobIds.map((jobId) => removeTemporaryAudio(jobId)));
  } else if (state.currentJob) {
    const cleanupPath = state.playbackMode === "quality-online"
      ? `/api/generate-stream/${state.currentJob}/ephemeral-audio`
      : `/api/generate-stream/${state.currentJob}/close`;
    fetch(cleanupPath, { method: state.playbackMode === "quality-online" ? "DELETE" : "POST" }).catch(() => {});
  }
  state.currentJob = "";
  if (state.audioContext && state.audioContext.state !== "closed") {
    try { await state.audioContext.close(); } catch (_) {}
  }
  state.audioContext = null;
  state.paused = false;
}

async function stopCurrentGeneration() {
  const button = $("stop-generation");
  if (button.disabled) return;
  button.disabled = true;
  const streamJobId = state.currentJob;
  const documentActive = Boolean(state.book && ["running", "stopping"].includes(state.book.state));
  state.stopRequested = true;
  try {
    if (state.playbackMode !== "quality-online" && streamJobId) {
      const path = state.playbackMode === "quality-online"
        ? `/api/generate-stream/${encodeURIComponent(streamJobId)}/ephemeral-audio`
        : `/api/generate-stream/${encodeURIComponent(streamJobId)}/close`;
      await jsonFetch(path, { method: state.playbackMode === "quality-online" ? "DELETE" : "POST" });
    }
    await stopPlayback(true);
    if (documentActive) {
      state.book = await jsonFetch(
        `/api/document-projects/${encodeURIComponent(state.book.id)}/stop`,
        { method: "POST" }
      );
      state.playbackMode = "quality-generating";
      $("generation-progress").classList.remove("hidden");
      $("generation-label").textContent = "正在服务端停止；当前完整段落保存后暂停";
      renderBook();
      setActionsEnabled(true);
      if (state.qualityContext) {
        state.qualityContext.stopDeadline = Date.now() + 120_000;
        state.pollTimer = window.setTimeout(pollQualityGeneration, 450);
      }
    } else {
      state.playbackMode = "";
      state.playingBlock = null;
      $("play-toggle").textContent = "▶";
      $("playback-mode").textContent = "生成已停止";
      $("playback-seed").textContent = "服务端任务已关闭";
      renderCurrentChapter();
      setActionsEnabled(Boolean(state.book));
      toast("当前实时生成已从服务端停止");
      state.stopRequested = false;
    }
  } catch (error) {
    state.stopRequested = false;
    setActionsEnabled(Boolean(state.book));
    toast(error.message, true);
  }
}

async function togglePlayback() {
  if (["quality", "quality-online"].includes(state.playbackMode) && state.qualityAudio.src) {
    if (state.qualityAudio.paused) {
      await state.qualityAudio.play();
      $("play-toggle").textContent = "Ⅱ";
    } else {
      state.qualityAudio.pause();
      $("play-toggle").textContent = "▶";
    }
    return;
  }
  if (state.playbackMode === "stream" && state.audioContext) {
    if (state.audioContext.state === "running") {
      await state.audioContext.suspend();
      $("play-toggle").textContent = "▶";
    } else {
      await state.audioContext.resume();
      $("play-toggle").textContent = "Ⅱ";
    }
    return;
  }
  await startSelectedListening();
}

function updatePlayer(segment, mode, seed) {
  $("player-title").textContent = currentChapter()?.title || state.book?.name || "正在播放";
  $("player-subtitle").textContent = `第 ${segment.index + 1} 块 · ${segment.text.slice(0, 38)}${segment.text.length > 38 ? "…" : ""}`;
  $("playback-mode").textContent = mode;
  $("playback-seed").textContent = seed == null ? "Seed 正在确定" : `Seed ${seed}`;
}

function scrollPlayingBlock() {
  document.querySelector(`[data-segment-index="${state.playingBlock}"]`)?.scrollIntoView({
    behavior: "smooth",
    block: "center",
  });
}

async function savePlayback(index, offset) {
  if (!state.book) return;
  const form = new FormData();
  form.append("segment_index", index);
  form.append("offset_seconds", offset);
  fetch(`/api/document-projects/${state.book.id}/playback`, { method: "POST", body: form }).catch(() => {});
}

function moveBlock(delta) {
  const segments = chapterSegments();
  if (!segments.length) return;
  const current = state.selectedBlock ?? segments[0].index;
  const position = Math.max(0, Math.min(segments.length - 1, segments.findIndex((item) => item.index === current) + delta));
  selectBlock(segments[position].index);
  scrollPlayingBlock();
}

function bindEvents() {
  $("book-file").onchange = (event) => openImport(event.target.files[0]);
  $("empty-book-file").onchange = (event) => openImport(event.target.files[0]);
  $("confirm-import").onclick = confirmImport;
  $("font-size").oninput = () => {
    document.documentElement.style.setProperty("--reader-font-size", `${$("font-size").value}px`);
    localStorage.setItem("qwen-reader-font-size", $("font-size").value);
  };
  $("playback-rate").onchange = () => {
    const rate = Number($("playback-rate").value || 1);
    state.qualityAudio.playbackRate = rate;
    localStorage.setItem("qwen-reader-playback-rate", String(rate));
    toast(`播放速度已设为 ${rate}×`);
  };
  $("previous-chapter").onclick = () => selectChapter(state.chapterIndex - 1);
  $("next-chapter").onclick = () => selectChapter(state.chapterIndex + 1);
  $("stream-listen").onclick = () => setListeningMode("stream");
  $("quality-generate").onclick = () => setListeningMode("quality");
  $("start-listening").onclick = startSelectedListening;
  $("generate-whole-book").onclick = generateWholeBook;
  $("book-progress-detail-button").onclick = () => {
    $("book-progress-detail").classList.toggle("hidden");
  };
  $("book-aac-bitrate").onchange = () => {
    localStorage.setItem("qwen-reader-aac-bitrate", selectedBookBitrate());
    updateBitrateLabel();
    setListeningMode(state.listeningMode, false);
  };
  $("stop-generation").onclick = stopCurrentGeneration;
  $("play-toggle").onclick = togglePlayback;
  $("previous-block").onclick = () => moveBlock(-1);
  $("next-block").onclick = () => moveBlock(1);
  $("toc-toggle").onclick = () => {
    const opened = document.body.classList.toggle("toc-open");
    $("toc-toggle").classList.toggle("active", opened);
    $("toc-toggle").setAttribute("aria-expanded", String(opened));
  };
  $("collapse-toc").onclick = () => {
    document.body.classList.remove("toc-open");
    $("toc-toggle").classList.remove("active");
    $("toc-toggle").setAttribute("aria-expanded", "false");
  };
  $("edit-mode-button").onclick = toggleEditMode;
  $("reader-settings-button").onclick = () => $("reader-settings-dialog").showModal();
  $("listening-controls-button").onclick = () => {
    updateListeningSummary();
    if (state.book) {
      renderWholeBookProgress();
      renderCurrentChapter();
    }
    $("listening-controls-dialog").showModal();
  };
  $("library-manage-button").onclick = async () => {
    state.editingBookId = "";
    state.pendingBookDeleteId = "";
    renderBookManager();
    $("library-manager-dialog").showModal();
    try {
      const data = await jsonFetch("/api/document-projects");
      state.books = data.projects || [];
      renderBooks();
    } catch (error) {
      toast(error.message, true);
    }
  };
  $("book-manager-tab-all").onclick = () => {
    state.bookManagerTab = "all";
    renderBookManager();
  };
  $("book-manager-tab-completed").onclick = () => {
    state.bookManagerTab = "completed";
    state.editingBookId = "";
    state.pendingBookDeleteId = "";
    renderBookManager();
  };
  $("manager-import-book").onclick = () => {
    $("library-manager-dialog").close();
    $("book-file").click();
  };
  $("theme-select").onchange = applyAppearance;
  $("accent-color").oninput = applyAppearance;
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.classList.contains("toc-open")) {
      document.body.classList.remove("toc-open");
      $("toc-toggle").classList.remove("active");
      $("toc-toggle").setAttribute("aria-expanded", "false");
    }
  });
  window.addEventListener("pagehide", () => {
    const ephemeralJobs = [...state.qualityJobs];
    for (const jobId of ephemeralJobs) {
      navigator.sendBeacon(
        `/api/generate-stream/${encodeURIComponent(jobId)}/ephemeral-audio/close`,
        new Blob([], { type: "application/octet-stream" })
      );
    }
    if (state.currentJob && !state.qualityJobs.has(state.currentJob)) {
      navigator.sendBeacon(
        `/api/generate-stream/${encodeURIComponent(state.currentJob)}/close`,
        new Blob([], { type: "application/octet-stream" })
      );
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      window.clearTimeout(state.pollTimer);
      return;
    }
    loadHealth().catch(() => {});
    loadServiceSettings().catch(() => {});
    if (state.qualityContext) {
      pollQualityGeneration();
    } else if (state.book && ["running", "stopping"].includes(state.book.state)) {
      refreshCurrentBook(true).catch(() => {});
    }
  });
}

async function init() {
  bindEvents();
  const savedFont = localStorage.getItem("qwen-reader-font-size");
  if (savedFont) $("font-size").value = savedFont;
  $("theme-select").value = localStorage.getItem("qwen-reader-theme") || "paper";
  $("accent-color").value = localStorage.getItem("qwen-reader-accent") || "#6555e8";
  $("book-aac-bitrate").value = localStorage.getItem("qwen-reader-aac-bitrate") || "80k";
  $("playback-rate").value = localStorage.getItem("qwen-reader-playback-rate") || "1";
  state.qualityAudio.playbackRate = Number($("playback-rate").value || 1);
  setListeningMode(localStorage.getItem("qwen-reader-listening-mode") || "stream", false);
  applyAppearance();
  updateBitrateLabel();
  document.documentElement.style.setProperty("--reader-font-size", `${$("font-size").value}px`);
  try {
    await Promise.all([loadHealth(), loadServiceSettings()]);
    await loadBooks();
    window.setInterval(() => {
      if (document.hidden) return;
      loadHealth();
      loadServiceSettings().catch(() => {});
    }, 5000);
    window.setInterval(() => {
      if (
        !document.hidden
        && state.book
        && ["running", "stopping"].includes(state.book.state)
        && !state.qualityContext
      ) {
        refreshCurrentBook(true).catch(() => {});
      }
    }, 1600);
    window.setInterval(() => {
      pollPlaybackControl().catch(() => {});
    }, 400);
  } catch (error) {
    toast(error.message, true);
    showEmpty();
  }
}

init();
