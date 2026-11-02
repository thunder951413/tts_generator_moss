// Qwen 声阅 — local novel reader.
//
// Layout of this file (classic script; reader-model.js must load first):
//   1. state + small utilities (fetch, toast, formatting)
//   2. audio priming for WebKit autoplay rules
//   3. server-side listening/lease bookkeeping
//   4. book model access (chapter index built once per full fetch)
//   5. library (summaries) + open/rename/delete
//   6. rendering (windowed chapter blocks, incremental status patches)
//   7. whole-book generation + /progress polling
//   8. player state machine: stream (prefetch), quality-online, saved AAC
//   9. playback position persistence
//  10. service/epoch polling, events, keyboard, init
const $ = (id) => document.getElementById(id);
const Model = globalThis.ReaderModel;

const PLAYER = Object.freeze({
  IDLE: "idle",
  STARTING: "starting",
  PLAYING: "playing",
  PAUSED: "paused",
  BETWEEN: "between",
  STOPPING: "stopping",
});

const state = {
  books: [],
  book: null,
  bookIndex: null,
  openSerial: 0,
  serviceSettings: null,
  serviceSettingsName: "正在读取服务设置",
  serviceSettingsSource: "default",
  serviceSettingsFingerprint: "",
  authRedirecting: false,
  bookRefreshSerial: 0,
  bookRefreshApplied: 0,
  // Player state machine. streamRunId is the run id: every local stop bumps it
  // and every async continuation re-checks it before touching shared state.
  player: PLAYER.IDLE,
  streamRunId: 0,
  run: null,
  stopped: false,
  paused: false,
  playbackMode: "",
  qualityContext: null,
  stopRequested: false,
  chapterIndex: 0,
  selectedBlock: null,
  playingBlock: null,
  resumePosition: null,
  playbackSeed: null,
  streamTimeline: [],
  streamOffset: 0,
  streamNextTime: 0,
  lastPositionTimestamp: 0,
  lastSavedPosition: null,
  listeningMode: "stream",
  playbackEpoch: null,
  playbackSessionId: "",
  playbackLeaseId: "",
  listeningSessionId: "",
  listeningRunId: 0,
  listeningEpoch: null,
  listeningHeartbeat: null,
  heartbeatInFlight: false,
  qualityJobs: new Set(),
  qualityPlaybackReject: null,
  qualityAudio: new Audio(),
  preloadAudio: new Audio(),
  audioContext: null,
  currentJob: "",
  editingBookId: "",
  pendingBookDeleteId: "",
  bookManagerTab: "all",
  editMode: false,
  pendingRerender: false,
  pollTimer: null,
  bookPollInFlight: false,
  progressUnsupported: false,
  serviceTimer: null,
  servicePollInFlight: false,
  controlPollInFlight: false,
  ticker: null,
  blockElements: new Map(),
  chapterButtons: [],
  renderWindow: null,
  rovingIndex: null,
  // Library / import / table of contents.
  libraryQuery: "",
  librarySort: "recent",
  managerQuery: "",
  collapsedSeries: new Set(),
  importFiles: [],
  importBusy: false,
  tocQuery: "",
  tocExpanded: new Set(),
  tocGroups: [],
  voiceCatalog: null,
  voiceDraftConfirm: false,
  followPlayback: true,
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
    const error = new Error(message || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function retryable(error) {
  if (error?.name === "AbortError") return false;
  const status = Number(error?.status || 0);
  return !status || status >= 500 || status === 429;
}

function fetchWithRetry(path, options = {}, tries = 3) {
  return Model.withRetry(() => jsonFetch(path, options), { tries, baseDelay: 400, shouldRetry: retryable });
}

// Fire-and-forget POST/DELETE: never blocks playback continuity.
function fireAndForget(path, method = "POST") {
  try {
    const pending = fetch(api(path), { method, keepalive: true });
    pending?.catch?.(() => {});
  } catch (_) {}
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function stoppedError() {
  return new DOMException("Stopped", "AbortError");
}

function storage(action, key, value) {
  try {
    if (action === "get") return localStorage.getItem(key);
    if (action === "set") localStorage.setItem(key, value);
    if (action === "remove") localStorage.removeItem(key);
  } catch (_) {}
  return null;
}

function toast(message, error = false) {
  const region = $("toast-region");
  if (!region) return;
  const item = document.createElement("div");
  item.className = `toast${error ? " error" : ""}`;
  item.textContent = message;
  region.appendChild(item);
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

function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function mediaUrl(bookId, path) {
  return `/api/document-projects/${encodeURIComponent(bookId)}/media?path=${encodeURIComponent(path)}`;
}

function newPlaybackSessionId() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID().replaceAll("-", "");
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}${Math.random().toString(36).slice(2)}`;
}

function playbackMediaUrl(bookId, path) {
  return `${mediaUrl(bookId, path)}&playback=1&session_id=${encodeURIComponent(state.playbackSessionId)}`;
}

function announce(message) {
  const region = $("reader-live");
  if (region && region.textContent !== message) region.textContent = message;
}

// ---------------------------------------------------------------------------
// 2. Audio priming (WebKit/Safari only allow audio started inside a gesture).

let silentWavURL = "";
function silentWav() {
  if (silentWavURL) return silentWavURL;
  try {
    const samples = 240;
    const bytes = new Uint8Array(44 + samples * 2);
    const view = new DataView(bytes.buffer);
    const text = (offset, value) => { for (let i = 0; i < value.length; i += 1) bytes[offset + i] = value.charCodeAt(i); };
    text(0, "RIFF"); view.setUint32(4, 36 + samples * 2, true); text(8, "WAVE");
    text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, 24000, true); view.setUint32(28, 48000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    text(36, "data"); view.setUint32(40, samples * 2, true);
    silentWavURL = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
  } catch (_) {
    silentWavURL = "";
  }
  return silentWavURL;
}

function ensureAudioContext() {
  if (state.audioContext && state.audioContext.state !== "closed") return state.audioContext;
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  try {
    state.audioContext = new AudioContextCtor({ latencyHint: "playback" });
  } catch (_) {
    state.audioContext = new AudioContextCtor();
  }
  return state.audioContext;
}

// Call synchronously from click/key handlers before any await.
function primeAudio() {
  try {
    const context = ensureAudioContext();
    if (context.state !== "running" && !state.paused) context.resume?.()?.catch?.(() => {});
  } catch (_) {}
  if (state.player !== PLAYER.IDLE) return;
  for (const audio of [state.qualityAudio, state.preloadAudio]) {
    if (audio.qwenPrimed) continue;
    const url = silentWav();
    if (!url) continue;
    audio.qwenPrimed = true;
    try {
      audio.src = url;
      const attempt = audio.play?.();
      attempt?.then?.(() => audio.pause(), () => {});
    } catch (_) {}
  }
}

function applyRate(audio) {
  const rate = Number($("playback-rate")?.value || 1);
  audio.playbackRate = rate;
  audio.defaultPlaybackRate = rate;
  audio.preservesPitch = true;
  audio.webkitPreservesPitch = true;
}

function resetAudioElement(audio) {
  if (!audio) return;
  audio.onended = null;
  audio.onerror = null;
  audio.onloadedmetadata = null;
  audio.oncanplaythrough = null;
  audio.qwenSegment = null;
  try { audio.pause?.(); } catch (_) {}
  // Keep the gesture-primed silent clip attached (WebKit unlock); drop real media.
  if (silentWavURL && audio.src === silentWavURL) return;
  try { audio.removeAttribute?.("src"); } catch (_) {}
  try { audio.load?.(); } catch (_) {}
}

// ---------------------------------------------------------------------------
// 3. Server bookkeeping: listening reservation, media lease, job cleanup.

function releaseMediaPlayback() {
  const leaseId = state.playbackLeaseId;
  state.playbackLeaseId = "";
  state.playbackSessionId = "";
  if (leaseId) fireAndForget(`/api/playback/${encodeURIComponent(leaseId)}/release`);
}

function releaseContinuousListening(runId = null) {
  if (runId != null && state.listeningRunId !== runId) return;
  const sessionId = state.listeningSessionId;
  window.clearInterval(state.listeningHeartbeat);
  state.listeningHeartbeat = null;
  state.listeningSessionId = "";
  state.listeningRunId = 0;
  state.listeningEpoch = null;
  if (sessionId) fireAndForget(`/api/listening/${encodeURIComponent(sessionId)}/release`);
}

async function reserveContinuousListening(runId) {
  const sessionId = newPlaybackSessionId();
  const reservation = await fetchWithRetry(
    `/api/listening/${encodeURIComponent(sessionId)}/heartbeat`,
    { method: "POST" }
  );
  if (!runActive(runId)) {
    fireAndForget(`/api/listening/${encodeURIComponent(sessionId)}/release`);
    throw stoppedError();
  }
  state.listeningSessionId = sessionId;
  state.listeningRunId = runId;
  state.listeningEpoch = Number(reservation.playback_epoch);
  if (Number.isFinite(state.listeningEpoch)) state.playbackEpoch = state.listeningEpoch;
  window.clearInterval(state.listeningHeartbeat);
  state.listeningHeartbeat = window.setInterval(() => heartbeatListening(sessionId, runId), 3000);
}

async function heartbeatListening(sessionId, runId) {
  if (state.heartbeatInFlight) return;
  if (state.listeningSessionId !== sessionId || state.listeningRunId !== runId) return;
  state.heartbeatInFlight = true;
  try {
    await Model.withRetry(() => jsonFetch(
      `/api/listening/${encodeURIComponent(sessionId)}/heartbeat?playback_epoch=${encodeURIComponent(state.listeningEpoch)}`,
      { method: "POST" }
    ), { tries: 3, baseDelay: 500, shouldRetry: retryable });
  } catch (error) {
    if (state.listeningSessionId === sessionId && state.listeningRunId === runId) {
      await interruptPlayback(error.status === 409 ? null : "connection");
    }
  } finally {
    state.heartbeatInFlight = false;
  }
}

// The server stopped us (or vanished): stop locally and explain why.
async function interruptPlayback(kind = null) {
  if (state.player === PLAYER.IDLE) return;
  const knownEpoch = state.playbackEpoch;
  stopLocal();
  state.stopped = true;
  let reason = kind;
  if (!reason) {
    try {
      const status = await jsonFetch("/api/playback/status");
      reason = Model.classifyEpochChange(knownEpoch, status.playback_epoch);
      if (reason === "none") reason = "expired";
    } catch (_) {
      reason = "connection";
    }
  }
  const messages = {
    stop: "后台已强制停止全部音频与运算",
    restart: "本地服务已重启，播放已停止；可从当前位置继续",
    connection: "与本地服务的连接中断，播放已停止",
    expired: "听书会话已过期（可能曾休眠），播放已停止",
  };
  $("playback-mode").textContent = reason === "restart" ? "服务已重启" : "播放已停止";
  toast(messages[reason] || messages.stop, reason !== "stop");
}

function cleanupJob(job) {
  if (!job?.id) return;
  state.qualityJobs.delete(job.id);
  if (job.kind === "ephemeral") {
    fireAndForget(`/api/generate-stream/${encodeURIComponent(job.id)}/ephemeral-audio`, "DELETE");
  } else {
    fireAndForget(`/api/generate-stream/${encodeURIComponent(job.id)}/close`);
  }
  state.run?.jobs.delete(job.id);
  if (state.currentJob === job.id) state.currentJob = "";
}

// ---------------------------------------------------------------------------
// 4. Book model access.

function setBook(book) {
  state.book = book;
  state.bookIndex = book ? Model.buildChapterIndex(book) : null;
  if (book && !Array.isArray(book.segments)) book.segments = [];
}

function bookSegments() {
  return state.book?.segments || [];
}

function segmentPosition(index) {
  const position = state.bookIndex?.position.get(Number(index));
  return position == null ? -1 : position;
}

function segmentByIndex(index) {
  const position = segmentPosition(index);
  return position < 0 ? null : bookSegments()[position];
}

function bookChapters() {
  return state.bookIndex?.chapters || [];
}

function currentChapter() {
  return bookChapters()[state.chapterIndex] || null;
}

function chapterSegments(chapterIndex = state.chapterIndex) {
  return state.bookIndex?.byChapter[chapterIndex] || [];
}

function chapterIndexForSegment(segmentIndex) {
  return state.bookIndex?.chapterOf.get(Number(segmentIndex)) ?? -1;
}

function remainingListeningSegments(startIndex = null) {
  if (!state.book) return [];
  const first = startIndex ?? state.selectedBlock ?? currentChapter()?.segment_start ?? 0;
  const position = segmentPosition(first);
  return position < 0 ? bookSegments().slice() : bookSegments().slice(position);
}

function completedListeningSegments(firstIndex) {
  const playable = [];
  const segments = bookSegments();
  for (let i = Math.max(0, segmentPosition(firstIndex)); i < segments.length; i += 1) {
    const segment = segments[i];
    if (segment.status !== "completed" || !segment.audio_file) break;
    playable.push(segment);
  }
  return playable;
}

function bookIsGenerating(book = state.book) {
  return ["running", "stopping"].includes(book?.state);
}

// ---------------------------------------------------------------------------
// Service health + settings.

async function loadHealth() {
  const pill = $("service-pill");
  try {
    const health = await jsonFetch("/api/health");
    const usable = ["ready", "degraded"].includes(health.state);
    pill.classList.toggle("ready", usable);
    pill.lastChild.textContent = usable
      ? `${health.active_profile_label || "TTS Metal"}${health.state === "degraded" ? " · 部分通道可用" : ""} · STT ${health.stt?.ready ? "就绪" : "未就绪"}`
      : `服务状态：${health.state}`;
  } catch (_) {
    pill.classList.toggle("ready", false);
    pill.lastChild.textContent = "本地服务未连接";
  }
}

function toBoolean(value, fallback = false) {
  if (value === undefined || value === null || value === "") return fallback;
  if (typeof value === "boolean") return value;
  return !["0", "false", "no", "off"].includes(String(value).toLowerCase());
}

function normalizeServiceSettings(shared = {}) {
  return Model.serviceToBookSettings(shared);
}

// The service default (what API callers and newly imported books use). It
// never changes the voice of an open book: books keep their own settings.
async function loadServiceSettings() {
  const payload = await jsonFetch("/api/service-settings");
  const nextSettings = normalizeServiceSettings(payload.settings || {});
  state.serviceSettings = nextSettings;
  state.serviceSettingsName = payload.name || nextSettings.voice_name;
  state.serviceSettingsSource = payload.source || "default";
  state.serviceSettingsFingerprint = JSON.stringify(nextSettings);
  updateListeningSummary();
  updateBitrateLabel();
}

// Settings the open book is read with (its own voice, never the service's).
function bookSettings(book = state.book) {
  return Model.bookVoiceSettings(book?.settings || {}, state.serviceSettings || {});
}

function bookVoiceLabel(book = state.book) {
  return String(book?.voice_label || book?.settings?.voice_name || "本书音色");
}

function servicePollDelay() {
  return bookIsGenerating() || state.player !== PLAYER.IDLE ? 5000 : 30000;
}

function scheduleServicePoll(delay = servicePollDelay()) {
  window.clearTimeout(state.serviceTimer);
  state.serviceTimer = window.setTimeout(runServicePoll, delay);
}

async function runServicePoll() {
  window.clearTimeout(state.serviceTimer);
  if (document.hidden) return;
  if (state.servicePollInFlight) return;
  state.servicePollInFlight = true;
  try {
    await Promise.allSettled([loadHealth(), loadServiceSettings()]);
  } finally {
    state.servicePollInFlight = false;
    scheduleServicePoll();
  }
}

// ---------------------------------------------------------------------------
// 5. Library.

function replaceBookSummary(book) {
  const summary = Model.summarizeBook(book);
  let found = false;
  state.books = state.books.map((item) => {
    if (item.id !== summary.id) return item;
    found = true;
    return summary;
  });
  if (!found) state.books.push(summary);
}

async function reloadLibrary() {
  const data = await jsonFetch("/api/document-projects");
  state.books = (data.projects || []).map(Model.summarizeBook);
  renderBooks();
}

async function loadBooks(preferredId = "") {
  const data = await jsonFetch("/api/document-projects");
  state.books = (data.projects || []).map(Model.summarizeBook);
  renderBooks();
  const remembered = preferredId || state.book?.id || storage("get", "qwen-reader-book") || "";
  const target = state.books.find((book) => book.id === remembered) || state.books[0];
  if (target) await openBook(target.id, Boolean(preferredId));
  else showEmpty();
}

function formatChars(count) {
  const value = Number(count || 0);
  return value >= 10000 ? `${(value / 10000).toFixed(value >= 100000 ? 0 : 1)} 万字` : `${value} 字`;
}

function bookCard(book) {
  const button = document.createElement("button");
  button.className = `book-card${state.book?.id === book.id ? " active" : ""}`;
  button.type = "button";
  const progress = Math.round(100 * Number(book.stats?.progress || 0));
  const initial = (book.name || "书").trim().slice(0, 1);
  const volumes = Number(book.volume_count || 0);
  const reading = book.reading_chapter_title
    ? `读到 ${book.reading_chapter_title}`
    : "尚未开始";
  button.innerHTML = `
    <span class="book-cover">${escapeHTML(initial)}</span>
    <span class="book-copy">
      <strong>${escapeHTML(book.name || "未命名小说")}</strong>
      <span class="book-reading">${escapeHTML(reading)}</span>
      <span>${Model.chapterCountOf(book)} 章${volumes > 1 ? ` · ${volumes} 卷` : ""} · ${formatChars(book.stats?.total_chars)} · 音频 ${progress}%${bookIsGenerating(book) ? " · 生成中" : ""}</span>
      <span class="book-progress"><i style="width:${progress}%"></i></span>
    </span>`;
  button.title = `${book.name || ""}${book.series ? `（${book.series}）` : ""}`;
  button.onclick = () => {
    closePanels();
    openBook(book.id).catch((error) => toast(error.message, true));
  };
  return button;
}

function renderBooks() {
  const list = $("book-list");
  list.innerHTML = "";
  updateSeriesOptions();
  if (!state.books.length) {
    list.innerHTML = '<p class="empty-copy">书架还是空的。点击右上角“＋”导入小说，或直接把文件拖进窗口。</p>';
    renderBookManager();
    return;
  }
  const entries = Model.groupLibrary(state.books, { query: state.libraryQuery, sort: state.librarySort });
  if (!entries.length) {
    list.innerHTML = `<p class="empty-copy">没有找到与“${escapeHTML(state.libraryQuery)}”相关的书。</p>`;
    renderBookManager();
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const entry of entries) {
    if (!entry.series) {
      fragment.appendChild(bookCard(entry.books[0]));
      continue;
    }
    const containsCurrent = entry.books.some((book) => book.id === state.book?.id);
    const collapsed = state.collapsedSeries.has(entry.series) && !containsCurrent && !state.libraryQuery;
    const group = document.createElement("section");
    group.className = `series-group${collapsed ? " collapsed" : ""}`;
    const header = document.createElement("button");
    header.type = "button";
    header.className = "series-header";
    header.setAttribute("aria-expanded", String(!collapsed));
    header.innerHTML = `
      <span class="series-chevron" aria-hidden="true">▾</span>
      <strong>${escapeHTML(entry.series)}</strong>
      <small>${entry.books.length} 本</small>`;
    header.onclick = () => {
      if (state.collapsedSeries.has(entry.series)) state.collapsedSeries.delete(entry.series);
      else state.collapsedSeries.add(entry.series);
      storage("set", "qwen-reader-collapsed-series", JSON.stringify([...state.collapsedSeries]));
      renderBooks();
    };
    group.appendChild(header);
    if (!collapsed) {
      const members = document.createElement("div");
      members.className = "series-books";
      for (const book of entry.books) members.appendChild(bookCard(book));
      group.appendChild(members);
    }
    fragment.appendChild(group);
  }
  list.appendChild(fragment);
  renderBookManager();
}

function updateSeriesOptions() {
  const datalist = $("series-options");
  if (!datalist) return;
  datalist.replaceChildren(...Model.knownSeries(state.books).map((name) => {
    const option = document.createElement("option");
    option.value = name;
    return option;
  }));
}

function bookStatusLabel(book) {
  if (book.state === "running") return "正在生成";
  if (book.state === "stopping") return "正在停止";
  if (book.state === "completed") return "音频已完成";
  if (book.state === "paused") return "已暂停";
  return "可阅读";
}

function managerButton(label, className, onclick, disabled = false, title = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `manager-action-button${className ? ` ${className}` : ""}`;
  button.textContent = label;
  button.disabled = disabled;
  if (title) button.title = title;
  button.onclick = onclick;
  return button;
}

function renderBookManager() {
  const list = $("managed-book-list");
  if (!list) return;
  const isCompleted = (book) => book.state === "completed" || book.final_audio;
  const completedCount = state.books.filter(isCompleted).length;
  const query = state.managerQuery.trim().toLowerCase();
  const displayedBooks = (state.bookManagerTab === "completed" ? state.books.filter(isCompleted) : state.books)
    .filter((book) => !query || [book.name, book.series].some((value) => String(value || "").toLowerCase().includes(query)))
    .sort((a, b) => Model.naturalCompare(a.series || a.name, b.series || b.name) || Model.naturalCompare(a.name, b.name));
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
    const completedTab = state.bookManagerTab === "completed";
    list.innerHTML = `
      <div class="manager-empty-state">
        <span>${completedTab ? "✓" : "书"}</span>
        <strong>${completedTab ? "还没有制作完成的整书音频" : "书架还是空的"}</strong>
        <p>${completedTab ? "完成整本书生成后，会集中显示在这里。" : "导入 TXT、Markdown 或 DOCX，系统会自动拆分章节。"}</p>
      </div>`;
    return;
  }
  for (const book of displayedBooks) {
    const progress = Math.round(100 * Number(book.stats?.progress || 0));
    const active = bookIsGenerating(book);
    const editing = state.editingBookId === book.id;
    const confirmingDelete = state.pendingBookDeleteId === book.id;
    const current = state.book?.id === book.id;
    const row = document.createElement("article");
    row.className = `managed-book-row${current ? " current" : ""}${confirmingDelete ? " confirming-delete" : ""}`;
    row.innerHTML = `
      <span class="manager-book-cover" aria-hidden="true">
        <span>${escapeHTML((book.name || "书").trim().slice(0, 1))}</span>
      </span>
      <div class="manager-book-copy">
        ${editing ? `
          <span class="manager-edit-fields">
            <input class="manager-rename-input" type="text" maxlength="120" value="${escapeHTML(book.name || "")}" aria-label="书名" placeholder="书名">
            <input class="manager-series-input" type="text" maxlength="80" list="series-options" value="${escapeHTML(book.series || "")}" aria-label="系列" placeholder="系列（可选，同系列归为一组）">
          </span>
        ` : `
          <span class="manager-book-title">
            <strong>${escapeHTML(book.name || "未命名小说")}</strong>
            ${book.series ? `<span class="manager-kind-badge series">${escapeHTML(book.series)}</span>` : ""}
            ${current ? '<span class="manager-kind-badge">当前阅读</span>' : ""}
            ${book.final_audio ? '<span class="manager-kind-badge completed">整书完成</span>' : ""}
          </span>
        `}
        <span class="manager-book-meta">
          <span>${Model.chapterCountOf(book)} 章${Number(book.volume_count || 0) > 1 ? ` · ${book.volume_count} 卷` : ""}</span>
          <span>${formatChars(book.stats?.total_chars)}</span>
          <span>${escapeHTML(book.voice_label || book.settings?.voice_name || "")}</span>
          <span>音频 ${progress}%</span>
          <span class="manager-book-state${active ? " active" : ""}">${bookStatusLabel(book)}</span>
        </span>
      </div>
      <div class="manager-book-actions"></div>`;
    const actions = row.querySelector(".manager-book-actions");
    if (editing) {
      const cancel = managerButton("取消", "", () => {
        state.editingBookId = "";
        renderBookManager();
      });
      const save = managerButton("保存", "primary", () => saveManagedBook(
        book.id,
        row.querySelector(".manager-rename-input").value,
        row.querySelector(".manager-series-input").value,
      ));
      actions.append(cancel, save);
      window.setTimeout(() => {
        const input = row.querySelector(".manager-rename-input");
        input?.focus();
        input?.select();
        for (const field of row.querySelectorAll("input")) {
          field.addEventListener("keydown", (event) => {
            if (event.key === "Enter") { event.preventDefault(); save.click(); }
            if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); cancel.click(); }
          });
        }
      }, 0);
    } else if (confirmingDelete) {
      actions.append(
        managerButton("取消", "", () => {
          state.pendingBookDeleteId = "";
          renderBookManager();
        }),
        managerButton("确认删除", "danger", () => deleteManagedBook(book.id))
      );
    } else {
      actions.append(
        managerButton(current ? "当前" : "打开", "primary", async () => {
          $("library-manager-dialog").close();
          await openBook(book.id).catch((error) => toast(error.message, true));
        }, current),
        managerButton("编辑", "", () => {
          state.editingBookId = book.id;
          state.pendingBookDeleteId = "";
          renderBookManager();
        }, false, "修改书名与系列"),
        managerButton("追加分卷", "", () => {
          $("library-manager-dialog").close();
          pickFilesForAppend(book.id);
        }, active, active ? "请先停止生成，再追加内容" : "把后续文件作为新的一卷追加到这本书"),
        managerButton("朗读设置", "", async () => {
          $("library-manager-dialog").close();
          if (state.book?.id !== book.id) await openBook(book.id).catch((error) => toast(error.message, true));
          if (state.book?.id === book.id) openBookVoiceDialog();
        }, active, active ? "生成中不能修改朗读设置" : "设置这本书的音色、模型与码率"),
        managerButton("删除", "danger-quiet", () => {
          state.pendingBookDeleteId = book.id;
          state.editingBookId = "";
          renderBookManager();
        }, active, active ? "请先停止生成，再删除这本书" : "删除书籍及其生成音频")
      );
    }
    list.appendChild(row);
  }
}

async function saveManagedBook(bookId, name, series) {
  const revised = String(name || "").replace(/\s+/g, " ").trim();
  if (!revised) return toast("书名不能为空", true);
  try {
    const updated = await jsonFetch(`/api/document-projects/${encodeURIComponent(bookId)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: revised, series: String(series || "").trim() }),
    });
    const patch = { name: updated.name || revised, series: updated.series || "" };
    state.books = state.books.map((book) => book.id === bookId ? { ...book, ...patch } : book);
    if (state.book?.id === bookId) {
      Object.assign(state.book, patch);
      renderBookHeader();
    }
    state.editingBookId = "";
    renderBooks();
    toast(patch.series ? `已保存《${patch.name}》· 系列“${patch.series}”` : `已保存《${patch.name}》`);
  } catch (error) {
    toast(error.message, true);
  }
}

function forgetBookStorage(bookId) {
  storage("remove", `qwen-reader-position-${bookId}`);
  storage("remove", `qwen-reader-chapter-${bookId}`);
  storage("remove", `qwen-reader-toc-${bookId}`);
  if (storage("get", "qwen-reader-book") === bookId) storage("remove", "qwen-reader-book");
}

async function deleteManagedBook(bookId) {
  const book = state.books.find((item) => item.id === bookId);
  try {
    if (state.book?.id === bookId) stopLocal({ save: false });
    await jsonFetch(`/api/document-projects/${encodeURIComponent(bookId)}`, { method: "DELETE" });
    forgetBookStorage(bookId);
    if (state.book?.id === bookId) {
      state.openSerial += 1;
      state.qualityContext = null;
      setBook(null);
    }
    state.pendingBookDeleteId = "";
    toast(`《${book?.name || "这本书"}》已从本机删除`);
    await loadBooks();
  } catch (error) {
    toast(error.message, true);
  }
}

async function openBook(bookId, remember = true) {
  const serial = ++state.openSerial;
  stopLocal();
  window.clearTimeout(state.pollTimer);
  // Generation/playback context always belongs to the previously open book.
  state.qualityContext = null;
  state.stopRequested = false;
  state.stopped = false;
  state.playbackMode = "";
  state.pendingRerender = false;
  const book = await jsonFetch(`/api/document-projects/${encodeURIComponent(bookId)}`);
  if (serial !== state.openSerial) return;
  setBook(book);
  state.renderWindow = null;
  state.lastSavedPosition = null;
  state.tocQuery = "";
  if ($("toc-search")) $("toc-search").value = "";
  state.tocExpanded = new Set();
  let cachedPosition = null;
  try { cachedPosition = JSON.parse(storage("get", `qwen-reader-position-${bookId}`)); } catch (_) {}
  const serverPosition = book.playback || {};
  state.resumePosition = Number(cachedPosition?.captured_at || 0) > Number(serverPosition.captured_at || 0)
    ? cachedPosition : serverPosition;
  state.chapterIndex = Math.min(
    Math.max(0, Number(storage("get", `qwen-reader-chapter-${bookId}`) || 0)),
    Math.max(0, bookChapters().length - 1)
  );
  const resumeIndex = state.resumePosition?.segment_index;
  state.selectedBlock = resumeIndex != null && segmentByIndex(resumeIndex)
    ? Number(resumeIndex)
    : Number(currentChapter()?.segment_start ?? 0);
  const resumeChapter = chapterIndexForSegment(state.selectedBlock);
  if (resumeChapter >= 0) state.chapterIndex = resumeChapter;
  state.rovingIndex = state.selectedBlock;
  if (remember) storage("set", "qwen-reader-book", bookId);
  replaceBookSummary(book);
  renderBooks();
  renderBook();
  resetPlayerLabels();
  if (bookIsGenerating()) {
    state.qualityContext = { bookId, title: book.name, passive: true };
    scheduleBookPoll(1500);
    scheduleServicePoll(5000);
  }
}

function showEmpty() {
  setBook(null);
  $("empty-state").classList.remove("hidden");
  $("reading-view").classList.add("hidden");
  $("book-title").textContent = "我的书架";
  $("book-kicker").textContent = "选择一本小说开始阅读";
  $("chapter-list").innerHTML = '<p class="empty-copy">打开小说后显示章节。</p>';
  state.chapterButtons = [];
  state.blockElements = new Map();
  setActionsEnabled(false);
  updateListeningSummary();
}

// ---------------------------------------------------------------------------
// 6. Rendering.

function renderBookHeader() {
  if (!state.book) return;
  $("book-title").textContent = state.book.name;
  const volumes = (state.book.volumes || []).length;
  $("book-kicker").textContent = [
    state.book.series,
    bookVoiceLabel(),
    `${bookChapters().length || 1} 章${volumes > 1 ? ` · ${volumes} 卷` : ""}`,
  ].filter(Boolean).join(" · ");
}

function renderDownloadLink() {
  const download = $("download-whole-book");
  if (state.book?.final_audio) {
    download.href = mediaUrl(state.book.id, state.book.final_audio);
    download.download = `${state.book.name}.m4a`;
    download.classList.remove("hidden");
  } else {
    download.removeAttribute("href");
    download.classList.add("hidden");
  }
}

function renderBook() {
  if (!state.book) return showEmpty();
  $("empty-state").classList.add("hidden");
  $("reading-view").classList.remove("hidden");
  renderBookHeader();
  renderDownloadLink();
  renderChapters();
  renderCurrentChapter();
  renderWholeBookProgress();
  updateBitrateLabel();
  setActionsEnabled(true);
}

function chapterSummaryText(chapterIndex) {
  const counts = Model.chapterCounts(chapterSegments(chapterIndex));
  return `${counts.total} 块 · ${counts.completed}/${counts.total} AAC`;
}

function chapterButton(chapter, position) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `chapter-button${position === state.chapterIndex ? " active" : ""}`;
  if (position === state.chapterIndex) button.setAttribute("aria-current", "true");
  button.innerHTML = `
    <span class="chapter-number">${String(position + 1).padStart(2, "0")}</span>
    <strong>${escapeHTML(chapter.title)}</strong>
    <small>${chapterSummaryText(position)}</small>`;
  button.onclick = () => selectChapter(position);
  state.chapterButtons[position] = button;
  return button;
}

// Volumes (or ranges of 100 chapters for long single-volume books) render as
// collapsible groups; only expanded groups create chapter buttons, so a
// 3,000-chapter book stays light.
function renderChapters() {
  const container = $("chapter-list");
  container.innerHTML = "";
  state.chapterButtons = [];
  const chapters = bookChapters();
  const fragment = document.createDocumentFragment();
  if (state.tocQuery) {
    const matches = Model.filterChapters(chapters, state.tocQuery);
    if (!matches.length) {
      container.innerHTML = `<p class="empty-copy">没有找到“${escapeHTML(state.tocQuery)}”。可输入章节标题中的文字或章号。</p>`;
      return;
    }
    for (const position of matches) fragment.appendChild(chapterButton(chapters[position], position));
    container.appendChild(fragment);
    return;
  }
  state.tocGroups = Model.buildTocGroups(chapters, state.book?.volumes || []);
  const currentGroup = Model.groupForChapter(state.tocGroups, state.chapterIndex);
  if (currentGroup) state.tocExpanded.add(currentGroup.key);
  for (const group of state.tocGroups) {
    if (group.title == null) {
      for (let position = group.start; position <= group.end; position += 1) {
        fragment.appendChild(chapterButton(chapters[position], position));
      }
      continue;
    }
    const expanded = state.tocExpanded.has(group.key);
    const section = document.createElement("section");
    section.className = `toc-group${expanded ? " expanded" : ""}${group === currentGroup ? " current" : ""}`;
    const header = document.createElement("button");
    header.type = "button";
    header.className = "toc-group-header";
    header.setAttribute("aria-expanded", String(expanded));
    let completed = 0;
    let total = 0;
    for (let position = group.start; position <= group.end; position += 1) {
      const counts = Model.chapterCounts(chapterSegments(position));
      completed += counts.completed;
      total += counts.total;
    }
    header.innerHTML = `
      <span class="series-chevron" aria-hidden="true">▾</span>
      <strong>${escapeHTML(group.title)}</strong>
      <small>${group.end - group.start + 1} 章${completed ? ` · 音频 ${Math.round((100 * completed) / Math.max(1, total))}%` : ""}</small>`;
    header.onclick = () => {
      if (state.tocExpanded.has(group.key)) state.tocExpanded.delete(group.key);
      else state.tocExpanded.add(group.key);
      renderChapters();
    };
    section.appendChild(header);
    if (expanded) {
      const body = document.createElement("div");
      body.className = "toc-group-body";
      for (let position = group.start; position <= group.end; position += 1) {
        body.appendChild(chapterButton(chapters[position], position));
      }
      section.appendChild(body);
    }
    fragment.appendChild(section);
  }
  container.appendChild(fragment);
}

function updateChapterButton(position) {
  const button = state.chapterButtons[position];
  const small = button?.querySelector?.("small");
  if (small) small.textContent = chapterSummaryText(position);
}

function setActiveChapterButton(previous, next) {
  const group = Model.groupForChapter(state.tocGroups, next);
  if (!state.tocQuery && group && group.title != null
    && (!state.tocExpanded.has(group.key) || group !== Model.groupForChapter(state.tocGroups, previous))) {
    // Entering another volume: expand it and mark it current.
    renderChapters();
    return;
  }
  const old = state.chapterButtons[previous];
  old?.classList.remove("active");
  old?.removeAttribute?.("aria-current");
  const current = state.chapterButtons[next];
  current?.classList.add("active");
  current?.setAttribute?.("aria-current", "true");
}

function renderChapterMeta() {
  const chapter = currentChapter();
  if (!chapter) return;
  const counts = Model.chapterCounts(chapterSegments());
  const volume = chapter.volume_index != null
    ? (state.book?.volumes || []).find((item) => Number(item.index) === Number(chapter.volume_index))
    : null;
  $("chapter-position").textContent = `${volume ? `${volume.title} · ` : ""}第 ${state.chapterIndex + 1} / ${bookChapters().length} 章`;
  $("chapter-title").textContent = chapter.title;
  $("chapter-progress").textContent = `${counts.total} 个文字块`;
  $("chapter-duration").textContent = counts.duration ? `已生成 ${formatDuration(counts.duration)} AAC` : "尚未生成音频";
  $("previous-chapter").disabled = state.chapterIndex <= 0;
  $("next-chapter").disabled = state.chapterIndex >= bookChapters().length - 1;
  $("selected-range").textContent = chapter.title;
  $("range-detail").textContent = `第 ${Number(chapter.segment_start) + 1}–${Number(chapter.segment_end) + 1} 块 · 共 ${counts.total} 块`;
}

function isEditingBlock() {
  const active = document.activeElement;
  return Boolean(state.editMode && active?.classList?.contains("reader-block"));
}

function renderCurrentChapter(focusIndex = null) {
  const chapter = currentChapter();
  if (!chapter) return;
  renderChapterMeta();
  if (isEditingBlock()) {
    // Never wipe text the user is typing; re-render after the edit is saved.
    state.pendingRerender = true;
    return;
  }
  state.pendingRerender = false;
  const segments = chapterSegments();
  const focus = focusIndex ?? state.playingBlock ?? state.selectedBlock;
  let focusPosition = segments.findIndex((segment) => segment.index === focus);
  if (focusPosition < 0) focusPosition = 0;
  state.renderWindow = Model.renderWindow(segments.length, focusPosition, focusIndex == null ? state.renderWindow : null);
  if (state.renderWindow.end > segments.length) state.renderWindow = Model.renderWindow(segments.length, focusPosition);
  renderCurrentChapterWindow(state.renderWindow);
}

function windowButton(direction, count) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `block-window-button ${direction}`;
  button.dataset.direction = direction;
  button.textContent = direction === "before" ? `显示前面 ${count} 个文字块` : `继续显示后面 ${count} 个文字块`;
  return button;
}

function extendRenderWindow(direction) {
  const segments = chapterSegments();
  if (!state.renderWindow) return;
  const step = 200;
  const main = document.querySelector(".reader-main");
  const before = main?.scrollHeight || 0;
  if (direction === "before") {
    state.renderWindow = { start: Math.max(0, state.renderWindow.start - step), end: state.renderWindow.end };
  } else {
    state.renderWindow = { start: state.renderWindow.start, end: Math.min(segments.length, state.renderWindow.end + step) };
  }
  renderCurrentChapterWindow(state.renderWindow);
  if (direction === "before" && main) main.scrollTop += (main.scrollHeight || 0) - before;
}

function renderCurrentChapterWindow(windowRange) {
  const segments = chapterSegments();
  const blocks = $("reader-blocks");
  const fragment = document.createDocumentFragment();
  state.blockElements = new Map();
  if (windowRange.start > 0) fragment.appendChild(windowButton("before", windowRange.start));
  for (let i = windowRange.start; i < windowRange.end; i += 1) fragment.appendChild(createBlockElement(segments[i]));
  if (windowRange.end < segments.length) fragment.appendChild(windowButton("after", segments.length - windowRange.end));
  blocks.replaceChildren(fragment);
  observeWindowEdge();
}

let windowObserver = null;
function observeWindowEdge() {
  if (typeof IntersectionObserver === "undefined") return;
  windowObserver?.disconnect();
  const after = document.querySelector(".block-window-button.after");
  if (!after) return;
  windowObserver = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) extendRenderWindow("after");
  }, { root: document.querySelector(".reader-main"), rootMargin: "600px 0px" });
  windowObserver.observe(after);
}

function createBlockElement(segment) {
  const block = document.createElement("p");
  block.className = "reader-block";
  block.dataset.segmentIndex = segment.index;
  block.dataset.originalText = segment.text || "";
  block.tabIndex = segment.index === state.rovingIndex ? 0 : -1;
  block.contentEditable = state.editMode ? "true" : "false";
  block.spellcheck = state.editMode;
  const text = document.createElement("span");
  text.className = "block-text";
  text.textContent = segment.text || "";
  block.appendChild(text);
  applyBlockStatus(block, segment);
  if (segment.index === state.selectedBlock) block.classList.add("selected");
  if (segment.index === state.playingBlock) block.classList.add("playing");
  state.blockElements.set(segment.index, block);
  return block;
}

function applyBlockStatus(block, segment) {
  block.classList.toggle("generating", segment.status === "generating" || segment.status === "encoding");
  block.classList.toggle("failed", segment.status === "failed");
  const label = Model.blockStatusLabel(segment, formatDuration);
  let status = block.querySelector?.(".block-state");
  if (!label) {
    status?.remove();
    return;
  }
  if (!status) {
    status = document.createElement("span");
    status.className = "block-state";
    status.contentEditable = "false";
    block.appendChild(status);
  }
  if (status.textContent !== label) status.textContent = label;
}

function updateBlockElement(index) {
  const block = state.blockElements.get(index);
  const segment = segmentByIndex(index);
  if (block && segment) applyBlockStatus(block, segment);
}

function setSelectedBlock(index, { focus = false } = {}) {
  const previous = state.selectedBlock;
  state.selectedBlock = index;
  state.blockElements.get(previous)?.classList.remove("selected");
  const block = state.blockElements.get(index);
  block?.classList.add("selected");
  setRovingBlock(index, focus);
}

function setRovingBlock(index, focus = false) {
  const old = state.blockElements.get(state.rovingIndex);
  if (old) old.tabIndex = -1;
  state.rovingIndex = index;
  const block = state.blockElements.get(index);
  if (block) {
    block.tabIndex = 0;
    if (focus) block.focus?.({ preventScroll: false });
  }
}

// Follow playback/navigation into another chapter. Returns true if it switched.
function activatePlaybackChapter(segmentIndex) {
  const chapterPosition = chapterIndexForSegment(segmentIndex);
  if (chapterPosition < 0 || chapterPosition === state.chapterIndex) return false;
  const previous = state.chapterIndex;
  state.chapterIndex = chapterPosition;
  storage("set", `qwen-reader-chapter-${state.book.id}`, String(chapterPosition));
  setActiveChapterButton(previous, chapterPosition);
  renderCurrentChapter(segmentIndex);
  return true;
}

function ensureBlockRendered(index) {
  if (chapterIndexForSegment(index) < 0) return null;
  if (!activatePlaybackChapter(index) && !state.blockElements.has(index)) renderCurrentChapter(index);
  return state.blockElements.get(index) || null;
}

function scrollToBlock(index) {
  const block = state.blockElements.get(index);
  block?.scrollIntoView?.({ behavior: "smooth", block: "center" });
}

function setPlayingBlock(index, { scroll = true, announceBlock = true } = {}) {
  const previous = state.playingBlock;
  if (previous === index) return;
  state.playingBlock = index;
  state.blockElements.get(previous)?.classList.remove("playing");
  if (index == null) return;
  ensureBlockRendered(index);
  state.blockElements.get(index)?.classList.add("playing");
  setSelectedBlock(index);
  const segment = segmentByIndex(index);
  if (segment) updatePlayerTitle(segment);
  if (scroll && state.followPlayback) scrollToBlock(index);
  if (announceBlock) announce(`正在朗读 ${currentChapter()?.title || ""} 第 ${index + 1} 块`);
}

function updatePlayerTitle(segment) {
  const text = String(segment.text || "").replace(/\s+/g, " ");
  $("player-title").textContent = currentChapter()?.title || state.book?.name || "正在播放";
  $("player-subtitle").textContent = `第 ${segment.index + 1} 块 · ${text.slice(0, 38)}${text.length > 38 ? "…" : ""}`;
}

function updatePlayer(segment, mode, seed) {
  updatePlayerTitle(segment);
  $("playback-mode").textContent = mode;
  $("playback-seed").textContent = seed == null ? "Seed 正在确定" : `Seed ${seed}`;
}

function resetPlayerLabels() {
  $("player-title").textContent = state.book?.name || "尚未开始播放";
  $("player-subtitle").textContent = "选择文字块或点击听书";
  $("playback-mode").textContent = state.listeningMode === "quality" ? "待播 · 高质量整块" : "待播 · 实时流式";
  $("playback-seed").textContent = "Seed —";
}

function renderWholeBookProgress() {
  if (!state.book) return;
  const segments = bookSegments();
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
  updateGenerationScopeLabel();
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

// ---------------------------------------------------------------------------
// Edit mode.

function editableBlockText(block) {
  const clone = block.cloneNode(true);
  clone.querySelectorAll(".block-state").forEach((item) => item.remove());
  const raw = clone.innerText ?? clone.textContent ?? "";
  return String(raw)
    .replace(/\r\n?/g, "\n")
    .replace(/[^\S\n]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

function restoreBlockText(block) {
  const segment = segmentByIndex(Number(block.dataset.segmentIndex));
  const text = document.createElement("span");
  text.className = "block-text";
  text.textContent = block.dataset.originalText || "";
  block.replaceChildren(text);
  if (segment) applyBlockStatus(block, segment);
}

async function saveSegmentEdit(block, segmentIndex) {
  const text = editableBlockText(block);
  const original = block.dataset.originalText || "";
  if (!text || text === original) {
    if (!text || block.querySelector(".block-text")?.textContent !== original) restoreBlockText(block);
    if (state.pendingRerender && !isEditingBlock()) renderCurrentChapter();
    return;
  }
  const bookId = state.book.id;
  block.contentEditable = "false";
  block.classList.add("generating");
  try {
    const updated = await jsonFetch(
      `/api/document-projects/${encodeURIComponent(bookId)}/segments/${segmentIndex}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }
    );
    if (state.book?.id !== bookId) return;
    if (!Array.isArray(updated?.segments)) {
      await refreshCurrentBook(true);
      toast(`第 ${segmentIndex + 1} 块已保存；旧 AAC 已失效`);
      return;
    }
    const sameShape = updated.segments.length === bookSegments().length;
    setBook(updated);
    reconcileResumePosition();
    replaceBookSummary(updated);
    toast(`第 ${segmentIndex + 1} 块已保存；旧 AAC 已失效`);
    if (sameShape && !state.pendingRerender) {
      const segment = segmentByIndex(segmentIndex);
      block.dataset.originalText = segment?.text || text;
      block.contentEditable = state.editMode ? "true" : "false";
      if (segment) applyBlockStatus(block, segment);
      updateChapterButton(state.chapterIndex);
      renderChapterMeta();
      renderWholeBookProgress();
    } else {
      renderChapters();
      renderCurrentChapter();
    }
  } catch (error) {
    toast(error.message, true);
    restoreBlockText(block);
    block.contentEditable = state.editMode ? "true" : "false";
    block.classList.remove("generating");
  }
}

function toggleEditMode() {
  if (!state.book) return;
  if (!state.editMode) stopLocal();
  // Blur first so a block being edited is saved while edit mode is still on.
  document.activeElement?.blur?.();
  state.editMode = !state.editMode;
  document.body.classList.toggle("edit-mode", state.editMode);
  const button = $("edit-mode-button");
  button.classList.toggle("active", state.editMode);
  button.setAttribute("aria-pressed", String(state.editMode));
  button.textContent = state.editMode ? "✓ 完成编辑" : "✎ 编辑正文";
  for (const block of state.blockElements.values()) {
    block.contentEditable = state.editMode ? "true" : "false";
    block.spellcheck = state.editMode;
  }
  setActionsEnabled(true);
  toast(state.editMode ? "编辑模式已开启；离开文字块时自动保存" : "已回到浏览模式");
}

// ---------------------------------------------------------------------------
// Navigation.

function selectChapter(index) {
  if (!state.book) return;
  const target = Math.max(0, Math.min(index, bookChapters().length - 1));
  if (state.player !== PLAYER.IDLE) stopLocal();
  const previous = state.chapterIndex;
  state.chapterIndex = target;
  state.selectedBlock = Number(currentChapter().segment_start);
  state.rovingIndex = state.selectedBlock;
  storage("set", `qwen-reader-chapter-${state.book.id}`, String(state.chapterIndex));
  // Reading a chapter moves the bookmark (also what "继续" resumes from).
  if (previous !== target) savePlayback(state.selectedBlock, 0);
  setActiveChapterButton(previous, target);
  renderCurrentChapter(state.selectedBlock);
  closePanels();
  document.querySelector(".reader-main")?.scrollTo?.({ top: 0, behavior: "smooth" });
  announce(`已打开 ${currentChapter().title}`);
}

// Click/Enter on a block: select it; if something is playing (or the block
// already has saved AAC) start listening from it.
function selectBlock(index) {
  setSelectedBlock(index);
  if (state.editMode) return;
  const segment = segmentByIndex(index);
  if (bookIsGenerating() && !(segment?.status === "completed" && segment.audio_file)) return;
  if (state.player !== PLAYER.IDLE) {
    if (index !== state.playingBlock) restartFrom(index);
  } else if (segment?.status === "completed" && segment.audio_file) {
    playQualitySegments(completedListeningSegments(index), false);
  }
}

function restartFrom(index) {
  if (state.player === PLAYER.STARTING || state.player === PLAYER.STOPPING) return;
  const mode = state.playbackMode;
  stopLocal({ save: false });
  setSelectedBlock(index);
  savePlayback(index, 0);
  const segment = segmentByIndex(index);
  if (segment?.status === "completed" && segment.audio_file) {
    playQualitySegments(completedListeningSegments(index), false);
  } else if (bookIsGenerating()) {
    toast("这一段还没有生成；整书生成期间只能播放已生成的段落", true);
  } else if (mode === "quality-online" || (mode === "quality" && state.listeningMode === "quality")) {
    startQualityChapter(index);
  } else {
    startStreamChapter(index);
  }
}

function moveBlock(delta) {
  if (!state.book) return;
  const segments = bookSegments();
  if (!segments.length) return;
  if (state.player === PLAYER.STARTING || state.player === PLAYER.STOPPING) return;
  const current = state.playingBlock ?? state.selectedBlock ?? segments[0].index;
  const position = Math.max(0, Math.min(segments.length - 1, segmentPosition(current) + delta));
  const target = segments[position].index;
  if (target === current && state.player === PLAYER.IDLE) return;
  if (state.player !== PLAYER.IDLE) {
    restartFrom(target);
    return;
  }
  ensureBlockRendered(target);
  setSelectedBlock(target);
  scrollToBlock(target);
  announce(`已选择第 ${target + 1} 块`);
}

// ---------------------------------------------------------------------------
// Controls.

function setActionsEnabled(enabled) {
  const generating = bookIsGenerating() || state.playbackMode === "quality-generating";
  const busy = generating || state.player !== PLAYER.IDLE;
  $("stream-listen").disabled = !enabled || generating;
  $("quality-generate").disabled = !enabled || generating;
  $("start-listening").disabled = !enabled || state.player !== PLAYER.IDLE || state.editMode;
  $("generate-whole-book").disabled = !enabled || busy;
  $("stop-generation").disabled = !busy;
  $("previous-block").disabled = !enabled;
  $("next-block").disabled = !enabled;
  $("stop-playback").disabled = state.player === PLAYER.IDLE;
  for (const id of ["open-book-voice", "book-bitrate-button", "adopt-service-voice"]) {
    const node = $(id);
    if (node) node.disabled = !enabled || generating;
  }
  updateRateControl();
}

function updateRateControl() {
  const select = $("playback-rate");
  if (!select) return;
  const locked = state.playbackMode === "stream";
  select.disabled = locked;
  select.title = locked ? "实时流式固定 1.0× 以保持音高；高质量 AAC 支持变速" : "播放速度";
  const hint = $("rate-hint");
  if (hint) hint.classList.toggle("hidden", !locked);
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
    : "当前：实时流式 · 固定 1.0×";
  if (state.player === PLAYER.IDLE) {
    $("playback-mode").textContent = qualitySelected ? "待播 · 高质量整块" : "待播 · 实时流式";
  }
  if (persist) storage("set", "qwen-reader-listening-mode", state.listeningMode);
}

// ▶ / 起播: blocks that already have saved AAC play from disk (instant, no
// GPU); synthesis starts only where the saved audio runs out.
function startSelectedListening() {
  primeAudio();
  $("listening-controls-dialog").close?.();
  if (state.player !== PLAYER.IDLE || !state.book) return;
  if (state.editMode) return toast("请先完成编辑，再开始听书");
  const start = listeningStartIndex();
  const saved = completedListeningSegments(start);
  if (saved.length) return playQualitySegments(saved, true);
  if (bookIsGenerating()) {
    return toast("这一段还没有生成。整书生成期间可以播放已生成的段落，或先停止生成再实时朗读", true);
  }
  return startLiveListening(start);
}

function listeningStartIndex() {
  // openBook selects the saved position; clicks and chapter changes move it.
  return state.selectedBlock ?? Number(currentChapter()?.segment_start ?? 0);
}

function startLiveListening(index) {
  if (state.listeningMode === "quality") return startQualityChapter(index);
  return startStreamChapter(index);
}

const IMPORT_PATTERN = /\.(txt|md|markdown|docx)$/i;

function importModeValue() {
  return document.querySelector('input[name="import-mode"]:checked')?.value || "merge";
}

function setImportMode(mode) {
  const radio = document.querySelector(`input[name="import-mode"][value="${mode}"]`);
  if (radio) radio.checked = true;
  updateImportMode();
}

function fileStem(file) {
  return String(file?.name || "").replace(/\.[^.]+$/, "").trim();
}

function openImport(fileList, { appendTo = "" } = {}) {
  const files = Model.sortFilesNaturally(Array.from(fileList || []).filter((file) => IMPORT_PATTERN.test(file.name)));
  const skipped = Array.from(fileList || []).length - files.length;
  if (skipped) toast(`已忽略 ${skipped} 个不支持的文件（支持 TXT、Markdown、DOCX）`, true);
  if (!files.length || state.importBusy) return;
  state.importFiles = files;
  const multiple = files.length > 1;
  const list = $("import-files");
  list.replaceChildren(...files.map((file, position) => {
    const row = document.createElement("div");
    row.className = "import-file";
    row.innerHTML = `
      <span class="file-icon">${multiple ? position + 1 : "文"}</span>
      <div><strong>${escapeHTML(file.name)}</strong><small>${(file.size / 1024).toFixed(1)} KB${multiple ? ` · 第 ${position + 1} 卷` : " · 将自动识别卷与章节"}</small></div>`;
    return row;
  }));
  list.classList.toggle("multiple", multiple);
  $("import-heading").textContent = multiple ? `导入 ${files.length} 个文件` : "加入书架";
  $("import-mode-merge-title").textContent = multiple ? "合并为一本书" : "作为新书";
  $("import-mode-merge-detail").textContent = multiple
    ? `按文件名顺序，每个文件作为一卷（共 ${files.length} 卷），目录按卷分组`
    : "自动识别卷与章节";
  $("import-mode-separate-option").classList.toggle("hidden", !multiple);
  const target = $("import-target");
  target.replaceChildren(...[...state.books]
    .sort((a, b) => (b.id === state.book?.id) - (a.id === state.book?.id) || Model.naturalCompare(a.name, b.name))
    .map((book) => {
      const option = document.createElement("option");
      option.value = book.id;
      option.textContent = `${book.series ? `${book.series} · ` : ""}${book.name}${bookIsGenerating(book) ? "（生成中，不可追加）" : ""}`;
      option.disabled = bookIsGenerating(book);
      return option;
    }));
  $("import-mode-append-option").classList.toggle("hidden", !state.books.length);
  const inferred = Model.inferSeriesName(files.map((file) => file.name));
  $("import-name").value = multiple ? (inferred || fileStem(files[0])) : fileStem(files[0]);
  $("import-series").value = "";
  $("import-series").dataset.inferred = inferred;
  if (appendTo) target.value = appendTo;
  setImportMode(appendTo ? "append" : "merge");
  fillVoiceSelect($("import-voice"), { includeKeep: false });
  loadVoiceCatalog().then(() => fillVoiceSelect($("import-voice"), { includeKeep: false })).catch(() => {});
  $("import-dialog").showModal();
}

function pickFilesForAppend(bookId) {
  const input = $("book-file");
  input.dataset.appendTo = bookId;
  input.click();
}

function updateImportMode() {
  const mode = importModeValue();
  const count = state.importFiles.length;
  $("import-name-field").classList.toggle("hidden", mode !== "merge");
  $("import-series-field").classList.toggle("hidden", mode === "append");
  $("import-target-field").classList.toggle("hidden", mode !== "append");
  $("import-voice-field").classList.toggle("hidden", mode === "append");
  $("import-block-field").classList.toggle("hidden", mode === "append");
  const series = $("import-series");
  if (mode === "separate" && !series.value) series.value = series.dataset.inferred || "";
  series.placeholder = mode === "separate" ? "系列名（这些书会归为一组）" : "同一系列的书会在书架中归为一组";
  $("import-note").textContent = mode === "append"
    ? "追加的内容使用目标书籍现有的音色与分块设置；已生成的音频会保留，整书文件需要重新合并。"
    : "导入后自动识别“第 N 卷 / 第 N 章 / Chapter N / 序章 / 番外”等目录结构。音色保存在书中，之后修改服务默认音色不会影响它。原文和音频仅保存在这台 Mac。";
  $("confirm-import").textContent = mode === "append"
    ? `追加 ${count} 个文件`
    : mode === "separate" ? `导入为 ${count} 本书` : count > 1 ? `合并 ${count} 个文件为一本书` : "解析并加入书架";
}

// ---------------------------------------------------------------------------
// Voices: the service default, saved presets and the reference library.

async function loadVoiceCatalog(force = false) {
  if (!force && state.voiceCatalog && Date.now() - state.voiceCatalog.at < 60000) return state.voiceCatalog;
  const [presets, voices] = await Promise.allSettled([jsonFetch("/api/presets"), jsonFetch("/api/voices")]);
  state.voiceCatalog = {
    at: Date.now(),
    presets: presets.status === "fulfilled" ? presets.value.presets || [] : [],
    voices: voices.status === "fulfilled" ? voices.value.voices || [] : [],
  };
  return state.voiceCatalog;
}

function fillVoiceSelect(select, { includeKeep }) {
  const previous = select.value;
  const option = (value, label) => {
    const node = document.createElement("option");
    node.value = value;
    node.textContent = label;
    return node;
  };
  const nodes = [];
  if (includeKeep) nodes.push(option("keep", `保持当前 · ${bookVoiceLabel()}`));
  nodes.push(option("service", `服务当前默认 · ${state.serviceSettingsName}`));
  const catalog = state.voiceCatalog || { presets: [], voices: [] };
  if (catalog.presets.length) {
    const group = document.createElement("optgroup");
    group.label = "我的预设";
    for (const preset of catalog.presets) group.appendChild(option(`preset:${preset.id}`, preset.name));
    nodes.push(group);
  }
  if (catalog.voices.length) {
    const group = document.createElement("optgroup");
    group.label = "音色库";
    for (const voice of catalog.voices) group.appendChild(option(`voice:${voice.id}`, voice.name));
    nodes.push(group);
  }
  select.replaceChildren(...nodes);
  const values = Array.from(select.options).map((item) => item.value);
  select.value = values.includes(previous) ? previous : values[0];
}

// → { settings, label } for a voice choice, on top of `base` settings.
function voiceChoice(value, base) {
  const catalog = state.voiceCatalog || { presets: [], voices: [] };
  if (value === "service" || !value) {
    return { settings: { ...settingsSnapshotForService() }, label: state.serviceSettingsName };
  }
  if (value.startsWith("preset:")) {
    const preset = catalog.presets.find((item) => `preset:${item.id}` === value);
    if (preset) {
      const settings = Model.serviceToBookSettings(preset.settings || {});
      if (!preset.settings?.voice_name) settings.voice_name = preset.name;
      return { settings, label: preset.name };
    }
  }
  if (value.startsWith("voice:")) {
    const voice = catalog.voices.find((item) => `voice:${item.id}` === value);
    if (voice) {
      return {
        settings: {
          ...base,
          reference_audio_path: voice.audio_path || voice.path,
          voice_name: voice.name,
          qwen_clone_mode: "xvec",
          qwen_reference_text: voice.transcript || "",
        },
        label: voice.name,
      };
    }
  }
  return { settings: { ...base }, label: bookVoiceLabel() };
}

function settingsSnapshotForService() {
  return { ...(state.serviceSettings || normalizeServiceSettings({})) };
}

// What the reader sends for realtime / online blocks: the open book's voice.
function settingsSnapshot() {
  return state.book ? bookSettings() : settingsSnapshotForService();
}

function hexToRGBA(hex, alpha) {
  const value = Number.parseInt(String(hex).replace("#", ""), 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

const FONT_STACKS = {
  serif: '"Songti SC", "STSong", "Noto Serif CJK SC", serif',
  sans: '-apple-system, "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", sans-serif',
  kai: '"Kaiti SC", "STKaiti", "KaiTi", "Songti SC", serif',
};
const CONTENT_WIDTHS = { narrow: "680px", normal: "880px", wide: "1080px" };

function applyTypography() {
  const root = document.documentElement.style;
  const size = $("font-size").value;
  const lineHeight = $("line-height").value;
  const family = FONT_STACKS[$("font-family-select").value] ? $("font-family-select").value : "serif";
  const width = CONTENT_WIDTHS[$("content-width-select").value] ? $("content-width-select").value : "normal";
  root.setProperty("--reader-font-size", `${size}px`);
  root.setProperty("--reader-line-height", lineHeight);
  root.setProperty("--reader-font-family", FONT_STACKS[family]);
  root.setProperty("--reader-width", CONTENT_WIDTHS[width]);
  $("font-size-value").textContent = size;
  $("line-height-value").textContent = Number(lineHeight).toFixed(1);
  storage("set", "qwen-reader-font-size", size);
  storage("set", "qwen-reader-line-height", lineHeight);
  storage("set", "qwen-reader-font-family", family);
  storage("set", "qwen-reader-content-width", width);
}

function applyAppearance() {
  const theme = $("theme-select").value;
  const accent = $("accent-color").value;
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.setProperty("--accent", accent);
  document.documentElement.style.setProperty("--accent-strong", accent);
  document.documentElement.style.setProperty("--accent-soft", hexToRGBA(accent, .11));
  $("accent-value").textContent = accent.toUpperCase();
  storage("set", "qwen-reader-theme", theme);
  storage("set", "qwen-reader-accent", accent);
}

function selectedBookBitrate() {
  return bookSettings().aac_bitrate || "80k";
}

function updateBitrateLabel() {
  const bitrate = selectedBookBitrate();
  $("quality-bitrate-label").textContent = `逐段非流式 · 双路预取 · AAC ${bitrate}`;
  const chip = $("book-bitrate-button");
  if (chip) chip.textContent = `AAC ${bitrate} · 设置`;
  updateListeningSummary();
}

function updateListeningSummary() {
  const service = state.serviceSettings || normalizeServiceSettings({});
  const settings = state.book ? bookSettings() : service;
  const set = (id, text) => { const node = $(id); if (node) node.textContent = text; };
  set("active-preset-name", state.book ? "本书音色" : "服务默认音色");
  set("active-voice-name", state.book ? bookVoiceLabel() : state.serviceSettingsName);
  set("active-generation-summary", Model.voiceSummary(settings));
  set("service-setting-name", state.serviceSettingsName);
  set("service-setting-detail", `${service.voice_name} · ${Model.modelLabel(service.model_profile)} · ${service.qwen_clone_mode === "icl" ? "ICL" : "X-vector"}`);
  set("settings-book-voice", state.book ? `《${state.book.name}》· ${bookVoiceLabel()}` : "未打开书籍");
  set("settings-book-voice-detail", state.book ? Model.voiceSummary(settings) : "打开一本书后可设置它的音色、模型与码率。");
  const settingsButton = $("settings-open-book-voice");
  if (settingsButton) settingsButton.disabled = !state.book;
  const drift = $("voice-drift-note");
  if (drift) {
    const differs = Boolean(state.book && state.serviceSettings && Model.voiceDiffers(settings, service));
    drift.classList.toggle("hidden", !differs);
    if (differs) {
      set("voice-drift-text", `服务默认音色现在是“${state.serviceSettingsName}”，本书仍使用“${bookVoiceLabel()}”，互不影响。`);
    }
  }
}

async function postImport(path, file, fields = {}) {
  const form = new FormData();
  form.append("document", file, file.name);
  for (const [key, value] of Object.entries(fields)) form.append(key, value);
  return jsonFetch(path, { method: "POST", body: form });
}

// Merge: the first file creates the book and the rest are appended as
// volumes. Separate: one book per file under a shared series. Append: every
// file becomes a new volume of an existing book.
async function confirmImport() {
  const files = state.importFiles;
  if (!files.length || state.importBusy) return;
  const mode = importModeValue();
  const button = $("confirm-import");
  const label = button.textContent;
  state.importBusy = true;
  button.disabled = true;
  const progress = (text) => { button.textContent = text; };
  const voice = voiceChoice($("import-voice").value, settingsSnapshotForService());
  const common = {
    max_chars: $("import-block-size").value,
    settings_json: JSON.stringify(voice.settings),
    voice_label: voice.label,
  };
  const series = $("import-series").value.trim();
  let done = 0;
  let focusId = "";
  try {
    if (mode === "append") {
      const targetId = $("import-target").value;
      if (!targetId) throw new Error("请选择要追加到的书");
      let updated = null;
      for (const file of files) {
        progress(`正在追加 ${done + 1}/${files.length}…`);
        updated = await postImport(`/api/document-projects/${encodeURIComponent(targetId)}/append`, file, {
          volume_title: fileStem(file),
        });
        done += 1;
      }
      focusId = targetId;
      toast(`已向《${updated?.name || "这本书"}》追加 ${done} 卷，现共 ${Model.chapterCountOf(updated)} 章`);
    } else if (mode === "separate") {
      for (const file of files) {
        progress(`正在导入 ${done + 1}/${files.length}…`);
        const book = await postImport("/api/document-projects", file, { ...common, name: fileStem(file), series });
        if (!focusId) focusId = book.id;
        done += 1;
      }
      toast(`已导入 ${done} 本书${series ? `，归入系列“${series}”` : ""}`);
    } else {
      progress(files.length > 1 ? `正在导入第 1/${files.length} 卷…` : "正在解析章节…");
      let book = await postImport("/api/document-projects", files[0], {
        ...common,
        name: $("import-name").value.trim(),
        series,
      });
      focusId = book.id;
      done = 1;
      for (const file of files.slice(1)) {
        progress(`正在导入第 ${done + 1}/${files.length} 卷…`);
        book = await postImport(`/api/document-projects/${encodeURIComponent(book.id)}/append`, file, {
          volume_title: fileStem(file),
        });
        done += 1;
      }
      const volumes = (book.volumes || []).length;
      toast(`《${book.name}》已加入书架：${Model.chapterCountOf(book)} 章${volumes > 1 ? `，${volumes} 卷` : ""}`);
    }
    $("import-dialog").close();
    state.importFiles = [];
  } catch (error) {
    const partial = done && done < files.length ? `已完成 ${done}/${files.length} 个文件；` : "";
    toast(`${partial}${error.message}`, true);
  } finally {
    state.importBusy = false;
    button.disabled = false;
    button.textContent = label;
    if (focusId && state.book?.id === focusId) {
      // Appended to the open book: refresh in place, keep playback running.
      await refreshCurrentBook(true).catch(() => {});
      await reloadLibrary().catch(() => {});
    } else if (focusId) {
      await loadBooks(focusId).catch((error) => toast(error.message, true));
    }
  }
}

// ---------------------------------------------------------------------------
// Book voice ("本书朗读设置"). Separate from the service default on purpose:
// changing the service (API) voice never rewrites a book, and changing a
// book never touches the service.

function bookVoiceDraft() {
  const current = bookSettings();
  const choice = voiceChoice($("book-voice-source").value, current);
  const draft = { ...choice.settings };
  draft.model_profile = $("book-voice-model").value;
  draft.seed = $("book-voice-random").checked ? -1 : Math.max(0, Math.min(999999, Number($("book-voice-seed").value || 0)));
  draft.aac_bitrate = $("book-aac-bitrate").value;
  if ($("book-voice-source").value === "keep") draft.language = current.language;
  return { settings: draft, label: choice.label };
}

function syncVoiceControls(settings) {
  $("book-voice-model").value = settings.model_profile === "qwen_1_7b" ? "qwen_1_7b" : "qwen_0_6b";
  const random = Number(settings.seed) < 0;
  $("book-voice-random").checked = random;
  $("book-voice-seed").disabled = random;
  $("book-voice-seed").value = random ? "" : String(Number(settings.seed ?? 1234));
}

function updateBookVoiceImpact() {
  if (!state.book) return;
  const current = bookSettings();
  const { settings } = bookVoiceDraft();
  const completed = bookSegments().filter((segment) => segment.status === "completed").length;
  const voiceChanged = Model.voiceDiffers(current, settings);
  const bitrateChanged = settings.aac_bitrate !== current.aac_bitrate;
  const impact = $("book-voice-impact");
  const save = $("book-voice-save");
  impact.classList.toggle("warning", voiceChanged && completed > 0);
  if (voiceChanged && completed > 0) {
    impact.textContent = `更换音色、模型或 Seed 会清除本书已生成的 ${completed} 个文字块音频，之后需要重新生成。阅读位置会保留。`;
  } else if (voiceChanged) {
    impact.textContent = "将使用新的音色朗读本书。尚未生成音频，无需清除。";
  } else if (bitrateChanged) {
    impact.textContent = completed
      ? "只改码率：已生成的文字块保留，合并整书时按新码率编码。"
      : "只改码率：之后生成的音频使用新码率。";
  } else {
    impact.textContent = "未修改。";
  }
  state.voiceDraftConfirm = false;
  save.textContent = voiceChanged && completed > 0 ? "清除音频并保存" : "保存";
  save.classList.toggle("danger", voiceChanged && completed > 0);
  save.disabled = !voiceChanged && !bitrateChanged && $("book-voice-source").value === "keep";
}

async function openBookVoiceDialog() {
  if (!state.book) return toast("请先打开一本书", true);
  if (bookIsGenerating()) return toast("整本书正在生成；请先停止生成再修改朗读设置", true);
  const current = bookSettings();
  $("book-voice-scope").textContent = `只影响《${state.book.name}》；服务默认音色（供 API 与新书使用）不会被修改。`;
  $("book-voice-current").textContent = bookVoiceLabel();
  $("book-voice-current-detail").textContent = Model.voiceSummary(current);
  fillVoiceSelect($("book-voice-source"), { includeKeep: true });
  $("book-voice-source").value = "keep";
  syncVoiceControls(current);
  $("book-aac-bitrate").value = current.aac_bitrate || "80k";
  updateBookVoiceImpact();
  $("reader-settings-dialog").close?.();
  $("listening-controls-dialog").close?.();
  $("book-voice-dialog").showModal();
  try {
    await loadVoiceCatalog();
    fillVoiceSelect($("book-voice-source"), { includeKeep: true });
    updateBookVoiceImpact();
  } catch (_) {
    // The service default and "keep" remain available.
  }
}

async function saveBookVoice({ settings, label }, { confirm = false } = {}) {
  const book = state.book;
  if (!book) return null;
  const response = await fetch(`/api/document-projects/${encodeURIComponent(book.id)}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings, voice_label: label, confirm_reset: confirm }),
  });
  if (response.status === 401) return jsonFetch("/api/document-projects"); // redirects to login
  const payload = await response.json().catch(() => ({}));
  if (response.status === 409 && payload.code === "reset_required") {
    const error = new Error(payload.detail || "需要确认清除已生成音频");
    error.resetRequired = Number(payload.completed_segments || 0);
    throw error;
  }
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  if (state.book?.id !== book.id) return payload;
  if (state.player !== PLAYER.IDLE && Model.voiceDiffers(bookSettings(), Model.bookVoiceSettings(payload.settings || {}))) {
    stopLocal();
  }
  setBook(payload);
  reconcileResumePosition();
  replaceBookSummary(payload);
  renderBook();
  renderBooks();
  updateBitrateLabel();
  return payload;
}

async function submitBookVoice() {
  const button = $("book-voice-save");
  const draft = bookVoiceDraft();
  const destructive = Model.voiceDiffers(bookSettings(), draft.settings)
    && bookSegments().some((segment) => segment.status === "completed");
  button.disabled = true;
  try {
    const result = await saveBookVoice(draft, { confirm: destructive });
    $("book-voice-dialog").close();
    toast(result?.audio_reset
      ? `本书改用“${draft.label}”朗读；旧音频已清除`
      : `本书朗读设置已保存 · ${Model.voiceSummary(bookSettings())}`);
  } catch (error) {
    if (error.resetRequired) {
      // The book gained audio meanwhile (another client); ask again.
      $("book-voice-impact").textContent = `${error.message}。再次点击以确认。`;
      button.textContent = "清除音频并保存";
    } else {
      toast(error.message, true);
    }
  } finally {
    button.disabled = false;
  }
}

async function adoptServiceVoice() {
  if (!state.book) return;
  if (bookIsGenerating()) return toast("整本书正在生成；请先停止生成", true);
  const completed = bookSegments().filter((segment) => segment.status === "completed").length;
  const settings = { ...settingsSnapshotForService(), aac_bitrate: selectedBookBitrate() };
  const button = $("adopt-service-voice");
  if (completed && button.dataset.confirm !== "1") {
    button.dataset.confirm = "1";
    button.textContent = `确认：清除 ${completed} 段音频并改用`;
    window.setTimeout(() => {
      button.dataset.confirm = "";
      button.textContent = "本书改用服务默认";
    }, 6000);
    return;
  }
  button.dataset.confirm = "";
  button.textContent = "本书改用服务默认";
  try {
    await saveBookVoice({ settings, label: state.serviceSettingsName }, { confirm: completed > 0 });
    toast(`本书已改用“${state.serviceSettingsName}”朗读`);
  } catch (error) {
    toast(error.message, true);
  }
}

// ---------------------------------------------------------------------------
// 7. Whole-book generation and incremental progress.

async function refreshCurrentBook(render = true) {
  if (!state.book) return null;
  const bookId = state.book.id;
  const serial = state.openSerial;
  const requestSerial = ++state.bookRefreshSerial;
  const refreshed = await jsonFetch(`/api/document-projects/${encodeURIComponent(bookId)}`);
  if (state.book?.id !== bookId || serial !== state.openSerial || requestSerial < state.bookRefreshApplied) {
    return state.book;
  }
  state.bookRefreshApplied = requestSerial;
  setBook(refreshed);
  reconcileResumePosition();
  replaceBookSummary(refreshed);
  if (render) {
    renderBook();
    renderBooks();
  }
  return state.book;
}

function applyProgress(progress) {
  const book = state.book;
  const before = JSON.stringify([book.state, book.stats?.progress, book.final_audio]);
  const result = Model.mergeProgress(book, progress, state.bookIndex);
  if (result.ignored) return false;
  if (result.needsFullReload) return true;
  reconcileResumePosition();
  const chapters = new Set();
  for (const index of result.changed) {
    updateBlockElement(index);
    const chapter = chapterIndexForSegment(index);
    if (chapter >= 0) chapters.add(chapter);
  }
  for (const chapter of chapters) updateChapterButton(chapter);
  if (chapters.has(state.chapterIndex)) renderChapterMeta();
  if (result.bookChanged || result.changed.length) {
    renderWholeBookProgress();
    renderDownloadLink();
  }
  if (before !== JSON.stringify([book.state, book.stats?.progress, book.final_audio])) {
    replaceBookSummary(book);
    renderBooks();
  }
  setActionsEnabled(true);
  return false;
}

function needsBookPoll() {
  return Boolean(state.book && (bookIsGenerating() || state.qualityContext?.bookId === state.book.id));
}

function scheduleBookPoll(delay = 1500) {
  window.clearTimeout(state.pollTimer);
  if (!needsBookPoll() || document.hidden) return;
  state.pollTimer = window.setTimeout(pollBookProgress, delay);
}

async function pollBookProgress() {
  window.clearTimeout(state.pollTimer);
  if (!state.book || state.bookPollInFlight) return;
  const book = state.book;
  const serial = state.openSerial;
  state.bookPollInFlight = true;
  let delay = 1500;
  try {
    let reload = state.progressUnsupported;
    if (!reload) {
      try {
        const since = Number(book.revision || 0);
        const progress = await fetchWithRetry(
          `/api/document-projects/${encodeURIComponent(book.id)}/progress?since=${since}`
        );
        if (serial !== state.openSerial || state.book !== book) return;
        reload = applyProgress(progress);
      } catch (error) {
        if (error.status !== 404 && error.status !== 405) throw error;
        state.progressUnsupported = true;
        reload = true;
      }
    }
    if (reload) {
      await refreshCurrentBook(true);
      if (serial !== state.openSerial) return;
    }
    finishGenerationIfDone();
  } catch (error) {
    if (serial !== state.openSerial) return;
    delay = 5000;
    $("generation-label").textContent = `进度暂时无法刷新：${error.message}`;
  } finally {
    state.bookPollInFlight = false;
    if (serial === state.openSerial) scheduleBookPoll(delay);
  }
}

function finishGenerationIfDone() {
  const context = state.qualityContext;
  if (!context || !state.book || context.bookId !== state.book.id) return;
  if (state.book.state === "stopping" && context.stopDeadline && Date.now() > context.stopDeadline) {
    state.qualityContext = null;
    state.stopRequested = false;
    state.playbackMode = state.playbackMode === "quality-generating" ? "" : state.playbackMode;
    setActionsEnabled(true);
    toast("服务端停止时间较长，已转为后台监控；已完成段落不会丢失", true);
    return;
  }
  if (bookIsGenerating()) return;
  state.qualityContext = null;
  if (state.playbackMode === "quality-generating") state.playbackMode = "";
  setActionsEnabled(true);
  const segments = bookSegments();
  const completed = segments.filter((segment) => segment.status === "completed").length;
  if (state.book.state === "error") {
    toast(state.book.message || "生成失败", true);
  } else if (state.stopRequested || (state.book.state === "paused" && completed < segments.length)) {
    toast(`已停止生成；已保存 ${completed}/${segments.length} 个完整 AAC 文字块`);
  } else if (state.book.final_audio) {
    toast(`《${context.title}》整本书已生成：${completed} 个 AAC 文字块，整书 M4A 已合并`);
  } else if (!context.passive) {
    toast(`《${context.title}》段落已完成，但整书文件尚未就绪`, true);
  }
  state.stopRequested = false;
}

async function beginQualityGeneration() {
  if (!state.book || !bookSegments().length) return;
  stopLocal();
  const book = state.book;
  const serial = state.openSerial;
  $("stream-listen").disabled = true;
  $("quality-generate").disabled = true;
  $("generate-whole-book").disabled = true;
  const range = generationRange();
  const form = new FormData();
  form.append("segment_start", range.start);
  form.append("segment_end", range.end);
  // The book's saved settings are authoritative: generation never silently
  // switches voice (and discards audio) because the service default changed.
  state.stopRequested = false;
  try {
    const updated = await jsonFetch(
      `/api/document-projects/${encodeURIComponent(book.id)}/start-selection`,
      { method: "POST", body: form }
    );
    if (serial !== state.openSerial) return;
    state.qualityContext = { bookId: book.id, title: book.name, passive: false };
    state.playbackMode = "quality-generating";
    setBook(updated);
    reconcileResumePosition();
    replaceBookSummary(updated);
    renderBook();
    renderBooks();
    $("generation-label").textContent = `正在生成：${range.label}`;
    scheduleBookPoll(1200);
    scheduleServicePoll(5000);
  } catch (error) {
    if (serial !== state.openSerial) return;
    state.qualityContext = null;
    state.playbackMode = "";
    setActionsEnabled(true);
    toast(error.message, true);
  }
}

function generateWholeBook() {
  return beginQualityGeneration();
}

// Segment range [start, end] for the "生成范围" choice, around the open chapter.
function generationRange(scope = $("generation-scope")?.value || "rest") {
  const segments = bookSegments();
  const chapters = bookChapters();
  const lastSegment = segments.length ? segments[segments.length - 1].index : 0;
  const chapter = currentChapter() || chapters[0];
  if (!chapter || scope === "book") return { start: 0, end: lastSegment, label: "整本书" };
  if (scope === "chapter") {
    return { start: Number(chapter.segment_start), end: Number(chapter.segment_end), label: chapter.title };
  }
  if (scope === "volume" && chapter.volume_index != null) {
    const members = chapters.filter((item) => Number(item.volume_index) === Number(chapter.volume_index));
    const volume = (state.book?.volumes || []).find((item) => Number(item.index) === Number(chapter.volume_index));
    return {
      start: Number(members[0].segment_start),
      end: Number(members[members.length - 1].segment_end),
      label: volume?.title || "当前卷",
    };
  }
  if (scope === "volume") return { start: 0, end: lastSegment, label: "整本书（未分卷）" };
  return { start: Number(chapter.segment_start), end: lastSegment, label: `${chapter.title} 至结尾` };
}

function updateGenerationScopeLabel() {
  if (!state.book) return;
  const range = generationRange();
  const segments = bookSegments();
  const first = segmentPosition(range.start);
  const last = segmentPosition(range.end);
  let pending = 0;
  for (let i = Math.max(0, first); i <= last && i < segments.length; i += 1) {
    if (segments[i].status !== "completed") pending += 1;
  }
  const volumeOption = $("generation-scope")?.querySelector('option[value="volume"]');
  if (volumeOption) volumeOption.disabled = currentChapter()?.volume_index == null;
  $("generate-button-title").textContent = pending ? `生成 ${range.label}` : `${range.label} 已全部生成`;
  $("generate-button-detail").textContent = pending
    ? `待生成 ${pending} 段 · 已生成的跳过 · 全书完成后自动合并 M4A`
    : "该范围的音频都已保存，可直接播放";
}

// ---------------------------------------------------------------------------
// 8. Player state machine.

function runActive(runId) {
  return !state.stopped && runId === state.streamRunId;
}

function assertRun(runId) {
  if (!runActive(runId)) throw stoppedError();
}

function setPlayerStatus(status) {
  state.player = status;
  const button = $("play-toggle");
  const showsPause = status === PLAYER.PLAYING || status === PLAYER.BETWEEN || status === PLAYER.STARTING;
  button.textContent = showsPause ? "Ⅱ" : "▶";
  button.setAttribute?.("aria-label", showsPause ? "暂停 (空格)" : "播放 (空格)");
  button.classList?.toggle("loading", status === PLAYER.STARTING || status === PLAYER.BETWEEN);
  document.body?.classList?.toggle("player-active", status !== PLAYER.IDLE);
  setActionsEnabled(Boolean(state.book));
}

function beginRun(mode) {
  stopLocal();
  const runId = ++state.streamRunId;
  state.stopped = false;
  state.paused = false;
  state.playbackMode = mode;
  state.playbackEpoch = null;
  state.streamTimeline = [];
  state.streamOffset = 0;
  state.streamNextTime = 0;
  state.run = { id: runId, mode, bookId: state.book?.id || "", jobs: new Map(), sources: new Set(),
    objectURLs: new Set(), aborts: new Set() };
  setPlayerStatus(PLAYER.STARTING);
  startTicker();
  return runId;
}

// Synchronous local teardown; server cleanup is fire-and-forget.
function stopLocal({ save = true } = {}) {
  if (save) flushPosition();
  const run = state.run;
  state.streamRunId += 1;
  state.run = null;
  stopTicker();
  if (state.qualityPlaybackReject) {
    const reject = state.qualityPlaybackReject;
    state.qualityPlaybackReject = null;
    reject(stoppedError());
  }
  resetAudioElement(state.qualityAudio);
  resetAudioElement(state.preloadAudio);
  if (run) {
    for (const controller of run.aborts) { try { controller.abort(); } catch (_) {} }
    for (const source of run.sources) {
      try { source.onended = null; source.stop?.(); } catch (_) {}
      try { source.disconnect?.(); } catch (_) {}
    }
    for (const url of run.objectURLs) URL.revokeObjectURL(url);
    for (const job of [...run.jobs.values()]) cleanupJob(job);
  }
  for (const jobId of [...state.qualityJobs]) cleanupJob({ id: jobId, kind: "ephemeral" });
  if (state.currentJob) cleanupJob({ id: state.currentJob, kind: "stream" });
  releaseContinuousListening();
  releaseMediaPlayback();
  state.currentJob = "";
  state.streamTimeline = [];
  state.streamNextTime = 0;
  const context = state.audioContext;
  if (context && context.state === "running") context.suspend?.()?.catch?.(() => {});
  state.paused = false;
  const wasMode = state.playbackMode;
  state.playbackMode = wasMode === "quality-generating" ? wasMode : "";
  setPlayingBlock(null);
  if (state.player !== PLAYER.IDLE) setPlayerStatus(PLAYER.IDLE);
  else updateRateControl();
}

async function stopPlayback(markStopped = true) {
  stopLocal();
  if (markStopped) state.stopped = true;
}

function finishRun(runId, message) {
  if (runId !== state.streamRunId) return;
  stopLocal({ save: false });
  $("playback-mode").textContent = message;
}

function trackJob(runId, id, kind) {
  if (!runActive(runId) || !state.run) {
    cleanupJob({ id, kind });
    throw stoppedError();
  }
  state.run.jobs.set(id, { id, kind });
  if (kind === "ephemeral") state.qualityJobs.add(id);
  state.currentJob = id;
}

function runAbortController(runId) {
  const controller = new AbortController();
  if (state.run?.id === runId) state.run.aborts.add(controller);
  return controller;
}

async function waitWhilePaused(runId) {
  while (state.paused) {
    assertRun(runId);
    await sleep(150);
  }
  assertRun(runId);
}

function startTicker() {
  stopTicker();
  state.ticker = window.setInterval(playerTick, 250);
}

function stopTicker() {
  window.clearInterval(state.ticker);
  state.ticker = null;
}

function playerTick() {
  if (state.player === PLAYER.IDLE) return;
  if (state.playbackMode === "stream") {
    const now = state.audioContext?.currentTime || 0;
    const position = currentPlaybackPosition();
    const head = state.streamTimeline[0];
    if (head && now >= head.start && head.segmentIndex != null && head.segmentIndex !== state.playingBlock) {
      setPlayingBlock(head.segmentIndex);
      savePlayback(head.segmentIndex, position?.offset ?? 0);
    }
    if (!state.paused && state.player !== PLAYER.STARTING) {
      const waiting = !head || now < head.start - 0.01;
      const next = waiting ? PLAYER.BETWEEN : PLAYER.PLAYING;
      if (state.player !== next) {
        setPlayerStatus(next);
        $("playback-mode").textContent = waiting ? "实时流式 · 正在生成下一块…" : "实时流式";
      }
    }
  }
  maybeSavePosition();
}

// --- Stream mode (gapless, one-block prefetch) ----------------------------

function streamForm(segment, seedOverride = null) {
  const settings = settingsSnapshot();
  const form = new FormData();
  form.append("mode", "voice_clone");
  form.append("language", settings.language || "auto");
  form.append("text", segment.text);
  form.append("codec_chunk_frames", String(settings.codec_chunk_frames));
  form.append("seed", String(seedOverride == null ? settings.seed : seedOverride));
  if (seedOverride != null) form.append("session_seed", String(seedOverride));
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
  form.append("use_service_settings", "0");
  if (state.listeningEpoch != null) form.append("expected_playback_epoch", String(state.listeningEpoch));
  return form;
}

async function startStreamJob(segment, seedOverride, runId, skipSeconds = 0) {
  assertRun(runId);
  const start = await jsonFetch("/api/generate-stream/start", { method: "POST", body: streamForm(segment, seedOverride) });
  trackJob(runId, start.job_id, "stream");
  if (Number.isFinite(Number(start.playback_epoch))) state.playbackEpoch = Number(start.playback_epoch);
  return { start, segment, segmentIndex: segment.index, skipSeconds, audioResponse: null };
}

function openAudioStream(job, runId) {
  const controller = runAbortController(runId);
  job.abort = controller;
  const pending = fetch(`/api/generate-stream/${encodeURIComponent(job.start.job_id)}/audio`, { signal: controller.signal });
  pending.catch?.(() => {});
  return pending;
}

function settle(promise) {
  return promise.then((value) => ({ value, error: null }), (error) => ({ value: null, error }));
}

function scheduleStreamChunk(audioContext, buffer, audioStart, segmentIndex) {
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  const startAt = Math.max(state.streamNextTime, audioContext.currentTime + 0.05);
  source.start(startAt);
  const run = state.run;
  if (run) {
    run.sources.add(source);
    source.onended = () => run.sources.delete(source);
  }
  state.streamTimeline.push({ start: startAt, end: startAt + buffer.duration, audioStart,
    audioEnd: audioStart + buffer.duration, rate: 1, segmentIndex });
  state.streamNextTime = startAt + buffer.duration;
  if (state.player === PLAYER.STARTING && !state.paused) {
    setPlayerStatus(PLAYER.PLAYING);
    $("playback-mode").textContent = "实时流式";
  }
}

// Read one job's PCM to the end, scheduling it on the shared timeline. Returns
// once the stream has fully arrived and the job reported "finished".
async function receiveStream(job, runId) {
  const { start } = job;
  assertRun(runId);
  state.currentJob = start.job_id;
  $("playback-seed").textContent = `Seed ${start.seed}${start.seed_mode === "random" ? " · 本次随机" : ""}`;
  const audioContext = ensureAudioContext();
  if (!state.paused && audioContext.state !== "running") await audioContext.resume?.()?.catch?.(() => {});
  assertRun(runId);
  const response = await (job.audioResponse || openAudioStream(job, runId));
  if (!response.ok || !response.body) throw new Error(await response.text?.() || "音频流无法打开");
  assertRun(runId);
  const playbackEpoch = Number(response.headers.get("X-Playback-Epoch"));
  if (Number.isFinite(playbackEpoch) && response.headers.get("X-Playback-Epoch") != null) state.playbackEpoch = playbackEpoch;
  // The /start response carries the profile's real PCM format and is race-free;
  // stream headers are only a fallback.
  const channels = Number(start.channels || response.headers.get("X-Audio-Channels") || 1);
  const sampleRate = Number(start.sample_rate || response.headers.get("X-Audio-Sample-Rate") || 24000);
  const reader = response.body.getReader();
  const frameBytes = 2 * channels;
  let remainder = new Uint8Array(0);
  let decodedFrames = 0;
  const skippedFrames = Math.floor(Math.max(0, Number(job.skipSeconds) || 0) * sampleRate);
  if (job.first !== false) state.streamOffset = skippedFrames / sampleRate;
  while (true) {
    let inactivityTimer;
    const timeout = new Promise((_, reject) => {
      inactivityTimer = window.setTimeout(() => {
        job.abort?.abort();
        reject(new Error("音频流超过 60 秒没有数据，已停止任务"));
      }, 60_000);
    });
    const { value, done } = await Promise.race([reader.read(), timeout]).finally(() => {
      window.clearTimeout(inactivityTimer);
    });
    if (done) break;
    assertRun(runId);
    let bytes = value;
    if (remainder.length) {
      const joined = new Uint8Array(remainder.length + value.length);
      joined.set(remainder);
      joined.set(value, remainder.length);
      bytes = joined;
      remainder = new Uint8Array(0);
    }
    const usable = bytes.length - (bytes.length % frameBytes);
    if (usable < bytes.length) remainder = bytes.slice(usable);
    if (!usable) continue;
    const view = new DataView(bytes.buffer, bytes.byteOffset, usable);
    const totalFrames = usable / frameBytes;
    const skip = Math.min(totalFrames, Math.max(0, skippedFrames - decodedFrames));
    const audioStart = (decodedFrames + skip) / sampleRate;
    decodedFrames += totalFrames;
    const frames = totalFrames - skip;
    if (!frames) continue;
    const buffer = audioContext.createBuffer(channels, frames, sampleRate);
    for (let channel = 0; channel < channels; channel += 1) {
      const output = buffer.getChannelData(channel);
      for (let frame = 0; frame < frames; frame += 1) {
        output[frame] = view.getInt16(((frame + skip) * channels + channel) * 2, true) / 32768;
      }
    }
    scheduleStreamChunk(audioContext, buffer, audioStart, job.segmentIndex ?? state.playingBlock);
  }
  assertRun(runId);
  if (remainder.length) throw new Error("音频流不完整，请重试当前文字块");
  const finalStatus = await fetchWithRetry(`/api/generate-stream/${encodeURIComponent(start.job_id)}/status`);
  assertRun(runId);
  if (finalStatus?.state === "closed") throw stoppedError();
  if (finalStatus?.state !== "finished") {
    throw new Error(finalStatus?.error || "语音生成未完整结束，请重试当前文字块");
  }
  return start;
}

async function waitForScheduledEnd(runId) {
  while (true) {
    assertRun(runId);
    const context = state.audioContext;
    if (!context || context.state === "closed") throw stoppedError();
    if (context.currentTime >= state.streamNextTime - 0.04) return;
    await sleep(100);
  }
}

// Single-block helper (kept for tests and simple callers).
async function playStreamingAudio(start, runId = state.streamRunId) {
  const job = { start, segmentIndex: state.playingBlock, skipSeconds: resumeOffset(state.playingBlock) };
  await receiveStream(job, runId);
  await waitForScheduledEnd(runId);
}

async function startStreamChapter(startIndex = null) {
  if (!state.book || state.player === PLAYER.STARTING) return;
  const segments = remainingListeningSegments(startIndex);
  if (!segments.length) return;
  const runId = beginRun("stream");
  $("playback-mode").textContent = "实时流式 · 正在起播";
  setPlayingBlock(segments[0].index);
  updatePlayer(segments[0], "实时流式 · 正在起播", null);
  let failed = false;
  let pending = null;
  let previous = null;
  try {
    await reserveContinuousListening(runId);
    let seed = state.resumePosition?.segment_index === segments[0].index ? state.resumePosition.seed : null;
    pending = settle(startStreamJob(segments[0], seed, runId, resumeOffset(segments[0].index)));
    for (let i = 0; i < segments.length; i += 1) {
      const outcome = await pending;
      pending = null;
      if (outcome.error) throw outcome.error;
      const job = outcome.value;
      job.first = i === 0;
      assertRun(runId);
      if (previous) {
        // The next block's audio request is queued for the playback lease;
        // releasing the previous job's lease now lets it start immediately.
        cleanupJob({ id: previous.start.job_id, kind: "stream" });
        previous = null;
      }
      await receiveStream(job, runId);
      seed = job.start.seed ?? seed;
      if (i + 1 < segments.length) {
        // The current block has fully arrived (GPU is free): prefetch the next
        // one so its PCM lands right after this block on the same timeline.
        previous = job;
        const nextSegment = segments[i + 1];
        pending = settle(startStreamJob(nextSegment, seed, runId, 0).then((next) => {
          next.audioResponse = openAudioStream(next, runId);
          return next;
        }));
      } else {
        previous = job;
      }
    }
    await waitForScheduledEnd(runId);
    if (previous) cleanupJob({ id: previous.start.job_id, kind: "stream" });
    saveNextPosition(segments[segments.length - 1].index);
  } catch (error) {
    failed = error?.name !== "AbortError";
    if (failed && runActive(runId)) {
      toast(error.message, true);
      // Let blocks that already arrived finish playing instead of cutting the
      // audible block off because the prefetched next block failed.
      try {
        await waitForScheduledEnd(runId);
      } catch {
        // stopped meanwhile
      }
    }
    if (runId === state.streamRunId) flushPosition();
  } finally {
    if (pending) pending.then((outcome) => outcome.value && cleanupJob({ id: outcome.value.start.job_id, kind: "stream" }));
    finishRun(runId, failed ? "播放失败，请重试当前块" : "本书播放完成");
  }
}

// --- Quality-online mode (non-streaming blocks rendered to temporary AAC) ---

function waitForJob(jobId, runId) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + 10 * 60 * 1000;
    const check = async () => {
      try {
        if (Date.now() > deadline) {
          reject(new Error("该文字块生成超过 10 分钟，任务已停止"));
          return;
        }
        const status = await fetchWithRetry(`/api/generate-stream/${encodeURIComponent(jobId)}/status`);
        if (!runActive(runId) || !state.qualityJobs.has(jobId)) {
          reject(stoppedError());
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

function playTemporaryAAC(prepared, runId) {
  return new Promise((resolve, reject) => {
    const audio = state.qualityAudio;
    const fail = (error) => {
      state.qualityPlaybackReject = null;
      reject(error);
    };
    state.qualityPlaybackReject = fail;
    state.playbackSeed = prepared.seed;
    audio.qwenPrimed = true;
    audio.src = prepared.objectURL;
    audio.qwenSegment = prepared.segment.index;
    applyRate(audio);
    seekSavedPosition(audio, prepared.segment.index);
    audio.onended = () => {
      state.qualityPlaybackReject = null;
      audio.qwenSegment = null;
      resolve();
    };
    audio.onerror = () => fail(new Error("临时 AAC 无法播放"));
    startAudioElement(audio, runId).catch(fail);
  });
}

// play() with WebKit autoplay handling: a NotAllowedError leaves the player
// paused so the next click (a user gesture) resumes it.
async function startAudioElement(audio, runId) {
  if (state.paused) return;
  try {
    await audio.play();
    if (runActive(runId)) setPlayerStatus(PLAYER.PLAYING);
  } catch (error) {
    if (!runActive(runId)) return;
    if (error?.name === "NotAllowedError") {
      state.paused = true;
      setPlayerStatus(PLAYER.PAUSED);
      $("playback-mode").textContent = "浏览器阻止了自动播放，请点击 ▶ 继续";
      return;
    }
    throw error;
  }
}

async function prepareQualityBlock(segment, runId) {
  const form = streamForm(segment, state.resumePosition?.segment_index === segment.index ? state.resumePosition.seed : null);
  form.set("streaming_generation", "0");
  form.set("qwen_non_streaming_mode", "1");
  form.set("ephemeral_audio", "1");
  assertRun(runId);
  const started = await jsonFetch("/api/generate-stream/start", { method: "POST", body: form });
  const jobId = started.job_id;
  trackJob(runId, jobId, "ephemeral");
  if (Number.isFinite(Number(started.playback_epoch))) state.playbackEpoch = Number(started.playback_epoch);
  try {
    const finished = await waitForJob(jobId, runId);
    assertRun(runId);
    const controller = runAbortController(runId);
    const response = await fetch(
      `/api/generate-stream/${encodeURIComponent(jobId)}/result-audio-aac?bitrate=${encodeURIComponent(selectedBookBitrate())}&playback=1`,
      { signal: controller.signal }
    );
    if (!response.ok) throw new Error(await response.text());
    const blob = await response.blob();
    assertRun(runId);
    const objectURL = URL.createObjectURL(blob);
    state.run?.objectURLs.add(objectURL);
    return { segment, jobId, seed: finished.seed ?? started.seed, seedMode: started.seed_mode, objectURL };
  } catch (error) {
    cleanupJob({ id: jobId, kind: "ephemeral" });
    throw error;
  }
}

function discardPrepared(prepared) {
  if (!prepared) return;
  URL.revokeObjectURL(prepared.objectURL);
  state.run?.objectURLs.delete(prepared.objectURL);
  cleanupJob({ id: prepared.jobId, kind: "ephemeral" });
}

function queueQualityBlock(segment, runId) {
  return prepareQualityBlock(segment, runId).then(
    (prepared) => ({ prepared, error: null }),
    (error) => ({ prepared: null, error })
  );
}

async function startQualityChapter(startIndex = null) {
  if (!state.book || state.player === PLAYER.STARTING) return;
  const segments = remainingListeningSegments(startIndex);
  if (!segments.length) return;
  const runId = beginRun("quality-online");
  setPlayingBlock(segments[0].index);
  updatePlayer(segments[0], "高质量非流式 · 正在完整生成", null);
  let pending = null;
  let failed = false;
  try {
    await reserveContinuousListening(runId);
    pending = queueQualityBlock(segments[0], runId);
    for (let index = 0; index < segments.length; index += 1) {
      const outcome = await pending;
      pending = null;
      if (outcome.error) throw outcome.error;
      const prepared = outcome.prepared;
      if (!runActive(runId)) {
        discardPrepared(prepared);
        break;
      }
      if (index + 1 < segments.length) {
        // While this complete block plays, the free lane prepares the next.
        pending = queueQualityBlock(segments[index + 1], runId);
      }
      try {
        await waitWhilePaused(runId);
        const segment = prepared.segment;
        setPlayingBlock(segment.index);
        updatePlayer(segment, `高质量 AAC ${selectedBookBitrate()} · 双路预取`, prepared.seed);
        $("playback-seed").textContent = `Seed ${prepared.seed}${prepared.seedMode === "random" ? " · 本次随机" : ""}`;
        await playTemporaryAAC(prepared, runId);
        saveNextPosition(segment.index);
        if (index + 1 < segments.length && runActive(runId)) {
          setPlayerStatus(PLAYER.BETWEEN);
          $("playback-mode").textContent = "高质量 AAC · 正在准备下一块…";
        }
      } finally {
        discardPrepared(prepared);
      }
    }
  } catch (error) {
    failed = error?.name !== "AbortError";
    if (runId === state.streamRunId) flushPosition();
    if (failed && runActive(runId)) toast(error.message, true);
  } finally {
    if (pending) pending.then((outcome) => discardPrepared(outcome.prepared));
    finishRun(runId, failed ? "播放失败，请重试当前块" : "本书播放完成");
  }
}

// --- Saved AAC mode (completed segments, gapless via a preload element) ----

function playQualitySegments(segments, resume = true) {
  if (!state.book || !segments.length || state.player === PLAYER.STARTING) return;
  primeAudio();
  const runId = beginRun("quality");
  if (!resume) savePlayback(segments[0].index, 0);
  state.playbackSessionId = newPlaybackSessionId();
  state.playbackLeaseId = `media:${state.playbackSessionId}`;
  state.run.queue = [...segments];
  $("playback-mode").textContent = "高质量 AAC";
  playNextQuality(runId, true);
}

function preloadNextQuality(runId) {
  const next = state.run?.id === runId ? state.run.queue[0] : null;
  const audio = state.preloadAudio;
  if (!next || audio.qwenSegment === next.index) return;
  audio.qwenPrimed = true;
  audio.qwenSegment = next.index;
  audio.preload = "auto";
  audio.src = playbackMediaUrl(state.book.id, next.audio_file);
  applyRate(audio);
  try { audio.load?.(); } catch (_) {}
}

function playNextQuality(runId = state.streamRunId, first = false) {
  if (!runActive(runId) || !state.run) return;
  if (!state.run.queue.length && state.book && state.playingBlock != null) {
    // Blocks finished by a running generation since this run started.
    const next = bookSegments()[segmentPosition(state.playingBlock) + 1];
    if (next) state.run.queue = completedListeningSegments(next.index);
  }
  const segment = state.run.queue.shift();
  if (!segment || !state.book) {
    const next = state.book && state.playingBlock != null
      ? bookSegments()[segmentPosition(state.playingBlock) + 1] : null;
    if (!next) {
      finishRun(runId, "本书播放完成");
    } else if (bookIsGenerating()) {
      finishRun(runId, "已播完已生成的部分");
      toast("后面的段落还在生成，生成后点 ▶ 继续");
    } else {
      // Saved audio ran out: continue seamlessly with live synthesis.
      finishRun(runId, "继续实时朗读");
      state.resumePosition = { segment_index: next.index, offset_seconds: 0 };
      startLiveListening(next.index);
    }
    return;
  }
  let audio = state.qualityAudio;
  if (!first && state.preloadAudio.qwenSegment === segment.index) {
    // Swap: the preloaded element becomes current, the old one preloads next.
    state.qualityAudio = state.preloadAudio;
    state.preloadAudio = audio;
    resetAudioElement(state.preloadAudio);
    audio = state.qualityAudio;
  } else {
    audio.qwenPrimed = true;
    audio.src = playbackMediaUrl(state.book.id, segment.audio_file);
    audio.qwenSegment = segment.index;
  }
  state.playbackSeed = segment.seed ?? null;
  applyRate(audio);
  if (first) seekSavedPosition(audio, segment.index);
  audio.onended = () => {
    if (!runActive(runId)) return;
    audio.qwenSegment = null;
    saveNextPosition(segment.index);
    playNextQuality(runId);
  };
  audio.onerror = () => {
    if (!runActive(runId)) return;
    toast("AAC 音频无法播放，请重试", true);
    finishRun(runId, "播放失败，请重试当前块");
  };
  audio.oncanplaythrough = () => preloadNextQuality(runId);
  setPlayingBlock(segment.index);
  updatePlayer(segment, "高质量 AAC", segment.seed);
  startAudioElement(audio, runId).catch((error) => {
    if (!runActive(runId)) return;
    toast(error.message, true);
    finishRun(runId, "播放失败，请重试当前块");
  });
  if ((audio.readyState || 0) >= 3) preloadNextQuality(runId);
  savePlayback(segment.index, first ? resumeOffset(segment.index) : 0);
}

// --- Pause / resume / stop -------------------------------------------------

function pausePlayback() {
  if (![PLAYER.PLAYING, PLAYER.BETWEEN].includes(state.player)) return;
  state.paused = true;
  if (state.playbackMode === "stream") state.audioContext?.suspend?.()?.catch?.(() => {});
  else state.qualityAudio.pause();
  setPlayerStatus(PLAYER.PAUSED);
  $("playback-mode").textContent = "已暂停";
  flushPosition();
}

function resumePlayback() {
  if (state.player !== PLAYER.PAUSED) return;
  state.paused = false;
  const runId = state.streamRunId;
  if (state.playbackMode === "stream") {
    state.audioContext?.resume?.()?.catch?.(() => {});
    setPlayerStatus(PLAYER.PLAYING);
    $("playback-mode").textContent = "实时流式";
    return;
  }
  const audio = state.qualityAudio;
  if (audio.qwenSegment != null) {
    // A block is loaded (paused mid-block): continue it inside this gesture.
    setPlayerStatus(PLAYER.PLAYING);
    $("playback-mode").textContent = state.playbackMode === "quality" ? "高质量 AAC" : `高质量 AAC ${selectedBookBitrate()}`;
    startAudioElement(audio, runId).catch((error) => {
      if (!runActive(runId)) return;
      toast(error.message, true);
      finishRun(runId, "播放失败，请重试当前块");
    });
  } else {
    // Paused between blocks: the run loop continues once `paused` clears.
    setPlayerStatus(PLAYER.BETWEEN);
  }
}

function togglePlayback() {
  primeAudio();
  switch (state.player) {
    case PLAYER.STARTING:
    case PLAYER.STOPPING:
      return;
    case PLAYER.PLAYING:
    case PLAYER.BETWEEN:
      pausePlayback();
      return;
    case PLAYER.PAUSED:
      resumePlayback();
      return;
    default:
      startSelectedListening();
  }
}

// Stop listening (any mode): ends playback and its live generation jobs, keeps the
// position so the next ▶ resumes there. Whole-book generation keeps running; the
// "停止当前生成" button controls that.
function stopListening() {
  if (state.player === PLAYER.IDLE) return;
  stopLocal();
  state.stopped = true;
  $("playback-mode").textContent = "已停止";
  setActionsEnabled(Boolean(state.book));
}

async function stopCurrentGeneration() {
  const button = $("stop-generation");
  if (button.disabled) return;
  button.disabled = true;
  const documentActive = Boolean(state.book && bookIsGenerating());
  const hadPlayer = state.player !== PLAYER.IDLE;
  state.stopRequested = true;
  stopLocal();
  state.stopped = true;
  if (hadPlayer) {
    $("playback-mode").textContent = "生成已停止";
    $("playback-seed").textContent = "服务端任务已关闭";
  }
  if (!documentActive) {
    state.stopRequested = false;
    setActionsEnabled(Boolean(state.book));
    if (hadPlayer) toast("当前实时生成已停止");
    return;
  }
  const book = state.book;
  const serial = state.openSerial;
  try {
    const updated = await jsonFetch(`/api/document-projects/${encodeURIComponent(book.id)}/stop`, { method: "POST" });
    if (serial !== state.openSerial) return;
    if (updated?.segments) setBook(updated);
    else Object.assign(state.book, updated || {});
    state.playbackMode = "quality-generating";
    $("generation-progress").classList.remove("hidden");
    $("generation-label").textContent = "正在服务端停止；当前完整段落保存后暂停";
    renderBook();
    if (!state.qualityContext) state.qualityContext = { bookId: book.id, title: book.name, passive: true };
    state.qualityContext.stopDeadline = Date.now() + 120_000;
    scheduleBookPoll(450);
  } catch (error) {
    state.stopRequested = false;
    setActionsEnabled(Boolean(state.book));
    toast(error.message, true);
  }
}

// ---------------------------------------------------------------------------
// 9. Playback position.

function reconcileResumePosition() {
  const position = state.book?.playback;
  if (position && (!state.resumePosition
      || Number(position.captured_at || 0) > Number(state.resumePosition.captured_at || 0))) {
    state.resumePosition = position;
    storage("set", `qwen-reader-position-${state.book.id}`, JSON.stringify(position));
  }
}

function resumeOffset(index) {
  return state.resumePosition?.segment_index === index
    ? Math.max(0, Number(state.resumePosition.offset_seconds) || 0) : 0;
}

function seekSavedPosition(audio, index) {
  const offset = resumeOffset(index);
  audio.onloadedmetadata = () => {
    if (offset > 0 && Number.isFinite(audio.duration)) {
      audio.currentTime = Math.min(offset, Math.max(0, audio.duration - 0.01));
    }
  };
}

function currentPlaybackPosition() {
  if (!state.book || state.playingBlock == null) return null;
  let offset = null;
  let index = state.playingBlock;
  if (["quality", "quality-online"].includes(state.playbackMode) && state.qualityAudio.src) {
    offset = Number(state.qualityAudio.currentTime);
  } else if (state.playbackMode === "stream" && state.audioContext && state.audioContext.state !== "closed") {
    const now = state.audioContext.currentTime || 0;
    while (state.streamTimeline.length && state.streamTimeline[0].end <= now) {
      const ended = state.streamTimeline.shift();
      state.streamOffset = ended.audioEnd;
      if (ended.segmentIndex != null) index = ended.segmentIndex;
    }
    const chunk = state.streamTimeline[0];
    if (chunk && now >= chunk.start) {
      offset = chunk.audioStart + Math.min(now - chunk.start, chunk.end - chunk.start) * (chunk.rate || 1);
      if (chunk.segmentIndex != null) index = chunk.segmentIndex;
    } else {
      offset = state.streamOffset;
    }
  }
  return Number.isFinite(offset) ? { index, offset: Math.max(0, offset) } : null;
}

function playbackForm(position) {
  const form = new FormData();
  form.append("segment_index", position.segment_index);
  form.append("offset_seconds", position.offset_seconds);
  form.append("captured_at", position.captured_at);
  form.append("seed", position.seed ?? -1);
  return form;
}

function rememberPlayback(index, offset) {
  if (!state.book) return null;
  state.lastPositionTimestamp = Math.max(Date.now(), state.lastPositionTimestamp + 1);
  const position = { segment_index: index, offset_seconds: Math.max(0, offset),
    captured_at: state.lastPositionTimestamp, seed: state.playbackSeed };
  state.resumePosition = position;
  state.book.playback = position;
  state.lastSavedPosition = { index, offset: position.offset_seconds, at: Date.now() };
  storage("set", `qwen-reader-position-${state.book.id}`, JSON.stringify(position));
  noteReadingProgress(index, position.captured_at);
  return position;
}

// Keep the shelf's "读到…" line in step with the open book.
function noteReadingProgress(index, capturedAt) {
  const summary = state.books.find((book) => book.id === state.book?.id);
  const chapter = bookChapters()[chapterIndexForSegment(index)];
  if (!summary || !chapter) return;
  summary.last_read_at = Number(capturedAt || Date.now()) / 1000;
  if (summary.reading_chapter_title === chapter.title) return;
  summary.reading_chapter_title = chapter.title;
  summary.reading_chapter_index = Number(chapter.index ?? 0);
  renderBooks();
}

// Never awaited by the player: the response is tiny and irrelevant to playback.
function savePlayback(index, offset) {
  const position = rememberPlayback(index, offset);
  if (!position) return Promise.resolve();
  const signal = typeof AbortSignal !== "undefined" && AbortSignal.timeout ? AbortSignal.timeout(5000) : undefined;
  return fetch(`/api/document-projects/${encodeURIComponent(state.book.id)}/playback`, {
    method: "POST", body: playbackForm(position), signal,
  }).then(() => {}, () => {});
}

// Called by the ticker: persist only while audio is actually advancing.
function maybeSavePosition() {
  if (state.player !== PLAYER.PLAYING || state.paused) return;
  const current = currentPlaybackPosition();
  if (!current || !Model.shouldSavePosition(state.lastSavedPosition, current, Date.now())) return;
  savePlayback(current.index, current.offset);
}

// Pause / stop / pagehide: save once if the position moved.
function flushPosition({ beacon = false } = {}) {
  if (state.player === PLAYER.IDLE || !state.book) return;
  const current = currentPlaybackPosition();
  if (!current || !Model.shouldSavePosition(state.lastSavedPosition, current, Date.now(), { force: true, minDelta: 0.25 })) return;
  if (beacon && navigator.sendBeacon) {
    const position = rememberPlayback(current.index, current.offset);
    navigator.sendBeacon(`/api/document-projects/${encodeURIComponent(state.book.id)}/playback`, playbackForm(position));
  } else {
    savePlayback(current.index, current.offset);
  }
}

function saveNextPosition(index) {
  const position = segmentPosition(index);
  const next = position >= 0 ? bookSegments()[position + 1] : null;
  return savePlayback(next?.index ?? index, next ? 0 : currentPlaybackPosition()?.offset || 0);
}

// ---------------------------------------------------------------------------
// 10. Polling, events, init.

async function pollPlaybackControl() {
  if (state.player === PLAYER.IDLE || state.controlPollInFlight || document.hidden) return;
  state.controlPollInFlight = true;
  const runId = state.streamRunId;
  try {
    const status = await jsonFetch("/api/playback/status");
    const epoch = Number(status.playback_epoch);
    if (!Number.isFinite(epoch) || runId !== state.streamRunId) return;
    if (state.playbackEpoch == null) {
      state.playbackEpoch = epoch;
      return;
    }
    const change = Model.classifyEpochChange(state.playbackEpoch, epoch);
    if (change !== "none") await interruptPlayback(change);
  } catch (_) {
    // Transient; heartbeats and stream errors handle a vanished service.
  } finally {
    state.controlPollInFlight = false;
  }
}

function closePanels() {
  document.body.classList.remove("toc-open", "library-open");
  $("toc-toggle").classList.remove("active");
  $("toc-toggle").setAttribute("aria-expanded", "false");
  $("library-toggle")?.setAttribute("aria-expanded", "false");
}

function isTypingTarget(target) {
  if (!target) return false;
  const tag = String(target.tagName || "").toLowerCase();
  return ["input", "textarea", "select"].includes(tag) || target.isContentEditable
    || target.contentEditable === "true";
}

function handleShortcut(event) {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.key === "Escape") {
    if (document.body.classList.contains("toc-open") || document.body.classList.contains("library-open")) {
      closePanels();
      event.preventDefault();
    }
    return;
  }
  if (isTypingTarget(event.target) || document.querySelector("dialog[open]")) return;
  const tag = String(event.target?.tagName || "").toLowerCase();
  switch (event.key) {
    case " ":
    case "Spacebar":
      if (tag === "button" || tag === "a") return;
      event.preventDefault();
      if (state.book) togglePlayback();
      break;
    case "s":
    case "S":
      event.preventDefault();
      stopListening();
      break;
    case "ArrowLeft":
    case "j":
    case "J":
      event.preventDefault();
      primeAudio();
      moveBlock(-1);
      break;
    case "ArrowRight":
    case "k":
    case "K":
      event.preventDefault();
      primeAudio();
      moveBlock(1);
      break;
    case "[":
      event.preventDefault();
      if (state.book && state.chapterIndex > 0) selectChapter(state.chapterIndex - 1);
      break;
    case "]":
      event.preventDefault();
      if (state.book && state.chapterIndex < bookChapters().length - 1) selectChapter(state.chapterIndex + 1);
      break;
    default:
  }
}

function bindBlockEvents() {
  const container = $("reader-blocks");
  container.addEventListener("click", (event) => {
    const more = event.target.closest?.(".block-window-button");
    if (more) {
      extendRenderWindow(more.dataset.direction);
      return;
    }
    const block = event.target.closest?.(".reader-block");
    if (!block) return;
    const index = Number(block.dataset.segmentIndex);
    if (state.editMode) {
      setSelectedBlock(index);
      return;
    }
    primeAudio();
    selectBlock(index);
  });
  container.addEventListener("keydown", (event) => {
    const block = event.target.closest?.(".reader-block");
    if (!block) return;
    const index = Number(block.dataset.segmentIndex);
    if (state.editMode) {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        restoreBlockText(block);
        block.blur();
      } else if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        block.blur();
      }
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      primeAudio();
      selectBlock(index);
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      const sibling = event.key === "ArrowDown" ? block.nextElementSibling : block.previousElementSibling;
      if (sibling?.classList.contains("reader-block")) {
        event.preventDefault();
        setSelectedBlock(Number(sibling.dataset.segmentIndex), { focus: true });
      }
    } else if (event.key === "Home" || event.key === "End") {
      const blocks = container.querySelectorAll(".reader-block");
      const target = event.key === "Home" ? blocks[0] : blocks[blocks.length - 1];
      if (target) {
        event.preventDefault();
        setSelectedBlock(Number(target.dataset.segmentIndex), { focus: true });
      }
    }
  });
  container.addEventListener("focusout", (event) => {
    const block = event.target.closest?.(".reader-block");
    if (block && state.editMode) saveSegmentEdit(block, Number(block.dataset.segmentIndex));
  });
}

function refreshOnFocus() {
  runServicePoll();
  if (needsBookPoll()) pollBookProgress();
}

function bindDropImport() {
  const overlay = $("drop-overlay");
  let depth = 0;
  const hasFiles = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");
  const hide = () => {
    depth = 0;
    overlay.classList.add("hidden");
  };
  window.addEventListener("dragenter", (event) => {
    if (!hasFiles(event) || document.querySelector("dialog[open]")) return;
    event.preventDefault();
    depth += 1;
    overlay.classList.remove("hidden");
  });
  window.addEventListener("dragover", (event) => {
    if (hasFiles(event)) event.preventDefault();
  });
  window.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) hide();
  });
  window.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    hide();
    if (document.querySelector("dialog[open]")) return;
    openImport(event.dataTransfer.files);
  });
}

function bindEvents() {
  const importFrom = (event) => {
    const files = Array.from(event.target.files || []);
    const appendTo = event.target.dataset.appendTo || "";
    event.target.value = "";
    event.target.dataset.appendTo = "";
    openImport(files, { appendTo });
  };
  $("book-file").onchange = importFrom;
  $("empty-book-file").onchange = importFrom;
  $("confirm-import").onclick = confirmImport;
  for (const radio of document.querySelectorAll('input[name="import-mode"]')) radio.onchange = updateImportMode;
  $("import-dialog").addEventListener("close", () => { if (!state.importBusy) state.importFiles = []; });
  for (const id of ["font-size", "line-height"]) $(id).oninput = applyTypography;
  for (const id of ["font-family-select", "content-width-select"]) $(id).onchange = applyTypography;
  $("follow-playback").onchange = () => {
    state.followPlayback = $("follow-playback").checked;
    storage("set", "qwen-reader-follow-playback", state.followPlayback ? "1" : "0");
  };
  let librarySearchTimer = null;
  $("library-search").oninput = () => {
    window.clearTimeout(librarySearchTimer);
    librarySearchTimer = window.setTimeout(() => {
      state.libraryQuery = $("library-search").value.trim();
      renderBooks();
    }, 120);
  };
  $("library-sort").onchange = () => {
    state.librarySort = $("library-sort").value;
    storage("set", "qwen-reader-library-sort", state.librarySort);
    renderBooks();
  };
  $("manager-search").oninput = () => {
    state.managerQuery = $("manager-search").value;
    renderBookManager();
  };
  let tocSearchTimer = null;
  $("toc-search").oninput = () => {
    window.clearTimeout(tocSearchTimer);
    tocSearchTimer = window.setTimeout(() => {
      state.tocQuery = $("toc-search").value.trim();
      if (state.book) renderChapters();
    }, 120);
  };
  $("toc-search").onkeydown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      state.tocQuery = $("toc-search").value.trim();
      const first = Model.filterChapters(bookChapters(), state.tocQuery, 1)[0];
      if (first != null) selectChapter(first);
    } else if (event.key === "Escape" && $("toc-search").value) {
      event.preventDefault();
      event.stopPropagation();
      $("toc-search").value = "";
      state.tocQuery = "";
      renderChapters();
    }
  };
  $("open-book-voice").onclick = openBookVoiceDialog;
  $("settings-open-book-voice").onclick = openBookVoiceDialog;
  $("book-bitrate-button").onclick = openBookVoiceDialog;
  $("adopt-service-voice").onclick = adoptServiceVoice;
  $("book-voice-save").onclick = submitBookVoice;
  $("book-voice-source").onchange = () => {
    const value = $("book-voice-source").value;
    syncVoiceControls(value === "keep" ? bookSettings() : voiceChoice(value, bookSettings()).settings);
    updateBookVoiceImpact();
  };
  $("book-voice-random").onchange = () => {
    $("book-voice-seed").disabled = $("book-voice-random").checked;
    if (!$("book-voice-random").checked && !$("book-voice-seed").value) $("book-voice-seed").value = "1234";
    updateBookVoiceImpact();
  };
  for (const id of ["book-voice-model", "book-aac-bitrate"]) $(id).onchange = updateBookVoiceImpact;
  $("book-voice-seed").oninput = updateBookVoiceImpact;
  bindDropImport();
  $("playback-rate").onchange = () => {
    const rate = Number($("playback-rate").value || 1);
    applyRate(state.qualityAudio);
    applyRate(state.preloadAudio);
    storage("set", "qwen-reader-playback-rate", String(rate));
    toast(`播放速度已设为 ${rate}×${state.listeningMode === "stream" ? "（实时流式固定 1.0×）" : ""}`);
  };
  $("previous-chapter").onclick = () => selectChapter(state.chapterIndex - 1);
  $("next-chapter").onclick = () => selectChapter(state.chapterIndex + 1);
  $("stream-listen").onclick = () => setListeningMode("stream");
  $("quality-generate").onclick = () => setListeningMode("quality");
  $("start-listening").onclick = startSelectedListening;
  $("generate-whole-book").onclick = generateWholeBook;
  $("generation-scope").onchange = () => {
    storage("set", "qwen-reader-generation-scope", $("generation-scope").value);
    updateGenerationScopeLabel();
  };
  $("book-progress-detail-button").onclick = () => $("book-progress-detail").classList.toggle("hidden");
  $("stop-generation").onclick = stopCurrentGeneration;
  $("play-toggle").onclick = togglePlayback;
  $("stop-playback").onclick = stopListening;
  $("previous-block").onclick = () => { primeAudio(); moveBlock(-1); };
  $("next-block").onclick = () => { primeAudio(); moveBlock(1); };
  $("toc-toggle").onclick = () => {
    document.body.classList.remove("library-open");
    const opened = document.body.classList.toggle("toc-open");
    $("toc-toggle").classList.toggle("active", opened);
    $("toc-toggle").setAttribute("aria-expanded", String(opened));
  };
  $("library-toggle").onclick = () => {
    document.body.classList.remove("toc-open");
    const opened = document.body.classList.toggle("library-open");
    $("library-toggle").setAttribute("aria-expanded", String(opened));
  };
  $("collapse-toc").onclick = closePanels;
  $("panel-scrim").onclick = closePanels;
  $("edit-mode-button").onclick = toggleEditMode;
  $("reader-settings-button").onclick = () => {
    updateListeningSummary();
    $("reader-settings-dialog").showModal();
  };
  $("listening-controls-button").onclick = () => {
    primeAudio();
    updateListeningSummary();
    if (state.book) {
      renderWholeBookProgress();
      renderChapterMeta();
    }
    $("listening-controls-dialog").showModal();
  };
  $("library-manage-button").onclick = async () => {
    state.editingBookId = "";
    state.pendingBookDeleteId = "";
    renderBookManager();
    $("library-manager-dialog").showModal();
    try {
      await reloadLibrary();
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
    $("book-file").dataset.appendTo = "";
    $("book-file").click();
  };
  $("theme-select").onchange = applyAppearance;
  $("accent-color").oninput = applyAppearance;
  bindBlockEvents();
  document.addEventListener("keydown", handleShortcut);
  window.addEventListener("pagehide", () => {
    flushPosition({ beacon: true });
    const beacon = (path) => navigator.sendBeacon?.(path, new Blob([], { type: "application/octet-stream" }));
    if (state.listeningSessionId) beacon(`/api/listening/${encodeURIComponent(state.listeningSessionId)}/release`);
    for (const jobId of state.qualityJobs) beacon(`/api/generate-stream/${encodeURIComponent(jobId)}/ephemeral-audio/close`);
    for (const job of state.run?.jobs.values() || []) {
      if (job.kind === "stream") beacon(`/api/generate-stream/${encodeURIComponent(job.id)}/close`);
    }
    if (state.playbackLeaseId) beacon(`/api/playback/${encodeURIComponent(state.playbackLeaseId)}/release`);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      flushPosition();
      window.clearTimeout(state.pollTimer);
      window.clearTimeout(state.serviceTimer);
      return;
    }
    refreshOnFocus();
  });
  window.addEventListener("focus", refreshOnFocus);
}

async function init() {
  bindEvents();
  const savedFont = storage("get", "qwen-reader-font-size");
  if (savedFont) $("font-size").value = savedFont;
  $("theme-select").value = storage("get", "qwen-reader-theme") || "paper";
  $("accent-color").value = storage("get", "qwen-reader-accent") || "#6555e8";
  $("line-height").value = storage("get", "qwen-reader-line-height") || "1.9";
  $("font-family-select").value = storage("get", "qwen-reader-font-family") || "serif";
  $("content-width-select").value = storage("get", "qwen-reader-content-width") || "normal";
  state.followPlayback = storage("get", "qwen-reader-follow-playback") !== "0";
  $("follow-playback").checked = state.followPlayback;
  state.librarySort = storage("get", "qwen-reader-library-sort") || "recent";
  $("generation-scope").value = storage("get", "qwen-reader-generation-scope") || "rest";
  $("library-sort").value = state.librarySort;
  try {
    state.collapsedSeries = new Set(JSON.parse(storage("get", "qwen-reader-collapsed-series") || "[]"));
  } catch (_) {
    state.collapsedSeries = new Set();
  }
  $("playback-rate").value = storage("get", "qwen-reader-playback-rate") || "1";
  applyRate(state.qualityAudio);
  applyRate(state.preloadAudio);
  setListeningMode(storage("get", "qwen-reader-listening-mode") || "stream", false);
  applyAppearance();
  updateBitrateLabel();
  applyTypography();
  try {
    await Promise.all([loadHealth(), loadServiceSettings().catch(() => {})]);
    await loadBooks();
  } catch (error) {
    toast(error.message, true);
    showEmpty();
  }
  scheduleServicePoll();
  window.setInterval(() => { pollPlaybackControl(); }, 1000);
}

init();
