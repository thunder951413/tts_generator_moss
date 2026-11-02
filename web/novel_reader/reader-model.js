// Pure, DOM-free helpers for the novel reader. Loaded as a classic script
// before reader.js (exposes globalThis.ReaderModel) and require()-able from
// node tests (module.exports).
(function attachReaderModel(root) {
  "use strict";

  const LARGE_CHAPTER_BLOCKS = 400;
  const RENDER_WINDOW_BEFORE = 120;
  const RENDER_WINDOW_AFTER = 220;

  // Group segments by chapter in one linear pass. Segments and chapters are
  // both ordered by index; a chapter covers [segment_start, segment_end].
  function buildChapterIndex(book) {
    const segments = Array.isArray(book?.segments) ? book.segments : [];
    const chapters = Array.isArray(book?.chapters) && book.chapters.length
      ? book.chapters
      : [{ index: 0, title: book?.name || "正文", segment_start: 0,
        segment_end: Math.max(0, segments.length - 1) }];
    const position = new Map();
    const chapterOf = new Map();
    const byChapter = chapters.map(() => []);
    let chapterCursor = 0;
    for (let i = 0; i < segments.length; i += 1) {
      const segment = segments[i];
      position.set(segment.index, i);
      while (
        chapterCursor < chapters.length - 1
        && segment.index > Number(chapters[chapterCursor].segment_end)
      ) {
        chapterCursor += 1;
      }
      const chapter = chapters[chapterCursor];
      if (
        segment.index >= Number(chapter.segment_start)
        && segment.index <= Number(chapter.segment_end)
      ) {
        byChapter[chapterCursor].push(segment);
        chapterOf.set(segment.index, chapterCursor);
      }
    }
    return { chapters, byChapter, position, chapterOf };
  }

  function chapterCounts(segments) {
    let completed = 0;
    let duration = 0;
    for (const segment of segments) {
      if (segment.status === "completed") {
        completed += 1;
        duration += Number(segment.duration_seconds || 0);
      }
    }
    return { total: segments.length, completed, duration };
  }

  const PROGRESS_FIELDS = [
    "revision", "state", "message", "stats", "active_segments", "current_segment",
    "final_audio", "updated_at",
  ];

  // Merge a /progress?since= payload into a full book in place.
  // Returns { needsFullReload, changed: number[], bookChanged }.
  function mergeProgress(book, progress, index = null) {
    if (!book || !progress || (progress.id && progress.id !== book.id)) {
      return { needsFullReload: false, changed: [], bookChanged: false, ignored: true };
    }
    if (
      progress.text_revision != null
      && book.text_revision != null
      && Number(progress.text_revision) !== Number(book.text_revision)
    ) {
      return { needsFullReload: true, changed: [], bookChanged: true };
    }
    if (progress.text_revision != null && book.text_revision == null) {
      book.text_revision = progress.text_revision;
    }
    let bookChanged = false;
    for (const field of PROGRESS_FIELDS) {
      if (!(field in progress)) continue;
      if (JSON.stringify(book[field]) !== JSON.stringify(progress[field])) bookChanged = true;
      book[field] = progress[field];
    }
    const positions = index?.position || null;
    const changed = [];
    let missing = false;
    for (const update of progress.segments || []) {
      const position = positions
        ? positions.get(update.index)
        : book.segments.findIndex((item) => item.index === update.index);
      const target = position == null || position < 0 ? null : book.segments[position];
      if (!target) {
        missing = true;
        continue;
      }
      let differs = false;
      for (const [key, value] of Object.entries(update)) {
        if (key === "text") continue;
        if (target[key] !== value && JSON.stringify(target[key]) !== JSON.stringify(value)) {
          differs = true;
          target[key] = value;
        }
      }
      if (differs) changed.push(update.index);
    }
    return { needsFullReload: missing, changed, bookChanged };
  }

  // A summary as served by GET /api/document-projects (no segments/chapters).
  function summarizeBook(book) {
    if (!book) return null;
    const { segments, chapters, volumes, sources, ...rest } = book;
    const summary = {
      ...rest,
      chapter_count: book.chapter_count ?? (Array.isArray(chapters) ? chapters.length : 1),
      volume_count: book.volume_count ?? (Array.isArray(volumes) ? volumes.length : 0),
    };
    if (summary.reading_chapter_title == null && Array.isArray(chapters) && chapters.length) {
      const index = Number(book.playback?.segment_index || 0);
      let reading = chapters[0];
      for (const chapter of chapters) {
        if (Number(chapter.segment_start) <= index) reading = chapter;
        else break;
      }
      summary.reading_chapter_title = book.playback?.captured_at ? String(reading.title || "") : "";
      summary.reading_chapter_index = Number(reading.index ?? 0);
    }
    if (summary.last_read_at == null && book.playback?.captured_at) {
      summary.last_read_at = Number(book.playback.captured_at) / 1000;
    }
    return summary;
  }

  function chapterCountOf(book) {
    if (book?.chapter_count != null) return Number(book.chapter_count) || 1;
    if (Array.isArray(book?.chapters)) return book.chapters.length || 1;
    return 1;
  }

  // The playback epoch starts at the service's boot time in milliseconds and a
  // global stop increments it by one, so a big jump means the service restarted.
  function classifyEpochChange(previous, next) {
    if (previous == null || next == null) return "none";
    const before = Number(previous);
    const after = Number(next);
    if (!Number.isFinite(before) || !Number.isFinite(after) || before === after) return "none";
    if (after < before || after - before > 1000) return "restart";
    return "stop";
  }

  // Decide whether a playback position is worth persisting.
  function shouldSavePosition(lastSent, next, nowMs, options = {}) {
    if (!next) return false;
    const minIntervalMs = options.minIntervalMs ?? 5000;
    const minDelta = options.minDelta ?? 0.75;
    if (!lastSent) return true;
    if (lastSent.index !== next.index) return true;
    if (Math.abs(Number(lastSent.offset) - Number(next.offset)) < minDelta) return false;
    if (options.force) return true;
    return nowMs - Number(lastSent.at || 0) >= minIntervalMs;
  }

  async function withRetry(task, options = {}) {
    const tries = Math.max(1, options.tries ?? 3);
    const baseDelay = options.baseDelay ?? 400;
    const sleep = options.sleep || ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    const shouldRetry = options.shouldRetry || ((error) => error?.name !== "AbortError");
    let lastError = null;
    for (let attempt = 0; attempt < tries; attempt += 1) {
      try {
        return await task(attempt);
      } catch (error) {
        lastError = error;
        if (attempt + 1 >= tries || !shouldRetry(error)) throw error;
        await sleep(baseDelay * (2 ** attempt));
      }
    }
    throw lastError;
  }

  // Which slice [start, end) of a chapter to render around a focus position.
  function renderWindow(total, focus, previous = null) {
    if (total <= LARGE_CHAPTER_BLOCKS) return { start: 0, end: total };
    const center = Math.max(0, Math.min(total - 1, Number(focus) || 0));
    if (previous && center >= previous.start && center < previous.end
        && previous.end <= total) {
      return { start: previous.start, end: previous.end };
    }
    return {
      start: Math.max(0, center - RENDER_WINDOW_BEFORE),
      end: Math.min(total, center + RENDER_WINDOW_AFTER),
    };
  }

  function blockStatusLabel(segment, formatDuration) {
    if (segment.status === "completed") return `AAC ${formatDuration(segment.duration_seconds)}`;
    if (segment.status === "generating") return "正在生成";
    if (segment.status === "encoding") return "正在转 AAC";
    if (segment.status === "failed") return "生成失败";
    return "";
  }


  // ---------------------------------------------------------------------------
  // Library: series grouping, search and sort.

  const collator = typeof Intl !== "undefined"
    ? new Intl.Collator("zh-Hans-CN", { numeric: true, sensitivity: "base" })
    : { compare: (a, b) => String(a).localeCompare(String(b)) };

  const CN_DIGITS = { 零: 0, 〇: 0, 一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9 };
  const CN_UNITS = { 十: 10, 百: 100, 千: 1000, 万: 10000 };

  function chineseNumber(text) {
    let total = 0;
    let section = 0;
    let digit = 0;
    for (const char of text) {
      if (char in CN_DIGITS) {
        digit = CN_DIGITS[char];
      } else if (char === "万") {
        total += (section + digit) * 10000;
        section = 0;
        digit = 0;
      } else {
        section += (digit || 1) * CN_UNITS[char];
        digit = 0;
      }
    }
    return total + section + digit;
  }

  // "第十二卷" → "第12卷", "（上）" → "（1）" so names sort in reading order.
  function sortKey(value) {
    return String(value || "")
      .replace(/[零〇一二两三四五六七八九十百千万]+/g, (match) => String(chineseNumber(match)))
      .replace(/([（(【\[\s_\-])([上中下])([）)】\]]?)(?=\.[^.]*$|$)/u, (_, open, part, close) => `${open}${"上中下".indexOf(part) + 1}${close}`);
  }

  function naturalCompare(a, b) {
    return collator.compare(sortKey(a), sortKey(b));
  }

  function lastActivity(book) {
    return Math.max(
      Number(book?.last_read_at || 0),
      Number(book?.created_at || 0),
    );
  }

  function bookMatches(book, query) {
    const needle = String(query || "").trim().toLowerCase();
    if (!needle) return true;
    return [book?.name, book?.series, book?.voice_label]
      .some((value) => String(value || "").toLowerCase().includes(needle));
  }

  // Books → display entries: { series: "" , books: [one] } for standalone
  // books and { series: "名字", books: [...] } for a series. A series sorts
  // by its most recent member; members sort naturally by name ("上" < "下",
  // "2" < "10").
  function groupLibrary(books, options = {}) {
    const sort = options.sort || "recent";
    const filtered = (books || []).filter((book) => bookMatches(book, options.query));
    const keyOf = (book) => {
      if (sort === "name") return String(book.name || "");
      if (sort === "added") return Number(book.created_at || 0);
      return lastActivity(book);
    };
    const groups = new Map();
    const entries = [];
    for (const book of filtered) {
      const series = String(book.series || "").trim();
      if (!series) {
        entries.push({ series: "", books: [book] });
        continue;
      }
      let group = groups.get(series);
      if (!group) {
        group = { series, books: [] };
        groups.set(series, group);
        entries.push(group);
      }
      group.books.push(book);
    }
    for (const group of groups.values()) group.books.sort((a, b) => naturalCompare(a.name, b.name));
    const groupKey = (entry) => {
      if (sort === "name") return entry.series || entry.books[0].name;
      return Math.max(...entry.books.map(keyOf));
    };
    entries.sort((a, b) => (sort === "name"
      ? naturalCompare(groupKey(a), groupKey(b))
      : groupKey(b) - groupKey(a)));
    return entries;
  }

  function knownSeries(books) {
    return [...new Set((books || []).map((book) => String(book.series || "").trim()).filter(Boolean))]
      .sort(naturalCompare);
  }

  // A shared name for files like "斗破苍穹-01.txt", "斗破苍穹 第二卷.txt" or
  // "三体（上）.txt": the common prefix minus volume markers and separators.
  function inferSeriesName(filenames) {
    const stems = (filenames || []).map((name) => String(name || "").replace(/\.[^.]+$/, "").trim())
      .filter(Boolean);
    if (!stems.length) return "";
    if (stems.length === 1) return stems[0];
    let prefix = stems[0];
    for (const stem of stems.slice(1)) {
      let i = 0;
      while (i < prefix.length && i < stem.length && prefix[i] === stem[i]) i += 1;
      prefix = prefix.slice(0, i);
    }
    const cleaned = prefix
      .replace(/(?:第[零〇一二三四五六七八九十百千两0-9]*|卷|vol\.?|volume|book|part|[（(【\[])$/i, "")
      .replace(/[\s_\-—–·.、:：(（【\[]+$/u, "")
      .replace(/[0-9]+$/, "")
      .replace(/[\s_\-—–·.、:：(（【\[]+$/u, "")
      .trim();
    return cleaned.length >= 2 ? cleaned : "";
  }

  function sortFilesNaturally(files) {
    return [...(files || [])].sort((a, b) => naturalCompare(a.name, b.name));
  }

  // ---------------------------------------------------------------------------
  // Table of contents: volumes, or ranges of chapters for very long books.

  const TOC_RANGE_SIZE = 100;
  const TOC_FLAT_LIMIT = 150;

  function chapterNumberLabel(position, rangeSize = TOC_RANGE_SIZE) {
    const start = Math.floor(position / rangeSize) * rangeSize;
    return `第 ${start + 1}–${start + rangeSize} 章`;
  }

  // → [{ key, title, start, end }] where start/end are chapter positions
  // (inclusive). title is null for a flat list.
  function buildTocGroups(chapters, volumes, options = {}) {
    const list = Array.isArray(chapters) ? chapters : [];
    const rangeSize = options.rangeSize || TOC_RANGE_SIZE;
    if (!list.length) return [];
    const titled = new Map((volumes || []).map((volume) => [Number(volume.index), volume]));
    const hasVolumes = titled.size > 0 && list.some((chapter) => chapter.volume_index != null);
    if (hasVolumes) {
      const groups = [];
      list.forEach((chapter, position) => {
        const volumeIndex = chapter.volume_index == null ? null : Number(chapter.volume_index);
        const key = volumeIndex == null ? `loose-${position}` : `volume-${volumeIndex}`;
        const last = groups[groups.length - 1];
        if (last && last.volumeIndex === volumeIndex && volumeIndex != null) {
          last.end = position;
        } else if (last && volumeIndex == null && last.volumeIndex == null) {
          last.end = position;
        } else {
          groups.push({
            key,
            volumeIndex,
            title: volumeIndex == null ? "未分卷" : String(titled.get(volumeIndex)?.title || `第 ${volumeIndex + 1} 卷`),
            start: position,
            end: position,
          });
        }
      });
      return groups;
    }
    if (list.length <= (options.flatLimit || TOC_FLAT_LIMIT)) {
      return [{ key: "all", volumeIndex: null, title: null, start: 0, end: list.length - 1 }];
    }
    const groups = [];
    for (let start = 0; start < list.length; start += rangeSize) {
      const end = Math.min(list.length, start + rangeSize) - 1;
      groups.push({ key: `range-${start}`, volumeIndex: null, title: `第 ${start + 1}–${end + 1} 章`, start, end });
    }
    return groups;
  }

  function groupForChapter(groups, position) {
    return (groups || []).find((group) => position >= group.start && position <= group.end) || null;
  }

  // Chapter positions whose title contains the query, or whose 1-based
  // number equals it ("12" finds the 12th chapter).
  function filterChapters(chapters, query, limit = 300) {
    const needle = String(query || "").trim().toLowerCase();
    if (!needle) return [];
    const number = /^\d+$/.test(needle) ? Number(needle) : null;
    const matches = [];
    (chapters || []).forEach((chapter, position) => {
      if (matches.length >= limit) return;
      if ((number != null && position + 1 === number)
        || String(chapter.title || "").toLowerCase().includes(needle)) {
        matches.push(position);
      }
    });
    return matches;
  }

  // ---------------------------------------------------------------------------
  // Voice settings. A book keeps its own voice; the service default only
  // seeds new books and API callers.

  function toBool(value, fallback = false) {
    if (value === undefined || value === null || value === "") return fallback;
    if (typeof value === "boolean") return value;
    return !["0", "false", "no", "off"].includes(String(value).toLowerCase());
  }

  // Service/preset settings (qwen_* keys) → book settings keys.
  function serviceToBookSettings(shared = {}) {
    return {
      model_profile: String(shared.model_profile || "qwen_0_6b"),
      reference_audio_path: String(shared.reference_audio_path || ""),
      voice_name: String(shared.voice_name || "服务默认音色"),
      language: String(shared.language || shared.qwen_language || "auto"),
      temperature: Number(shared.qwen_temperature ?? shared.temperature ?? 0.9),
      top_p: Number(shared.qwen_top_p ?? shared.top_p ?? 1),
      top_k: Number(shared.qwen_top_k ?? shared.top_k ?? 50),
      repetition_penalty: Number(shared.qwen_repetition_penalty ?? shared.repetition_penalty ?? 1.05),
      max_new_tokens: Number(shared.qwen_max_new_tokens ?? shared.max_new_tokens ?? 2048),
      codec_chunk_frames: Number(shared.qwen_chunk_size ?? shared.codec_chunk_frames ?? 8),
      seed: Number(shared.qwen_seed ?? shared.seed ?? 1234),
      qwen_clone_mode: String(shared.qwen_clone_mode || "xvec"),
      qwen_reference_text: String(shared.qwen_reference_text || ""),
      qwen_non_streaming_mode: false,
      qwen_append_silence: toBool(shared.qwen_append_silence, true),
      qwen_min_new_tokens: Number(shared.qwen_min_new_tokens ?? 2),
      aac_bitrate: String(shared.qwen_aac_bitrate || shared.aac_bitrate || "80k"),
    };
  }

  // A saved book's settings in the shape the reader sends back (the seed the
  // user chose, not the one resolved for a random run).
  function bookVoiceSettings(saved = {}, fallback = {}) {
    const base = { ...serviceToBookSettings(fallback), ...serviceToBookSettings(saved) };
    const configured = saved.configured_seed ?? (saved.seed_mode === "random" ? -1 : saved.seed);
    if (configured != null) base.seed = Number(configured);
    if (saved.language) base.language = String(saved.language);
    return base;
  }

  const VOICE_KEYS = [
    "model_profile", "reference_audio_path", "qwen_clone_mode", "qwen_reference_text",
    "seed", "temperature", "top_p", "top_k", "repetition_penalty", "qwen_append_silence", "language",
  ];

  // Would switching from book settings `a` to `b` change the generated voice?
  function voiceDiffers(a, b) {
    const norm = (settings, key) => {
      const value = settings?.[key];
      if (typeof value === "string") return value.trim();
      if (typeof value === "boolean") return value;
      if (value == null) return null;
      const number = Number(value);
      return Number.isFinite(number) ? Math.round(number * 1e6) / 1e6 : value;
    };
    return VOICE_KEYS.some((key) => norm(a, key) !== norm(b, key));
  }

  function modelLabel(profile) {
    return profile === "qwen_1_7b" ? "1.7B 高质量" : "0.6B 极速";
  }

  function voiceSummary(settings = {}) {
    const seed = Number(settings.seed);
    return `${modelLabel(settings.model_profile)} · ${seed < 0 ? "随机 Seed" : `Seed ${seed}`} · AAC ${settings.aac_bitrate || "80k"}`;
  }

  const api = {
    LARGE_CHAPTER_BLOCKS,
    TOC_RANGE_SIZE,
    naturalCompare,
    groupLibrary,
    knownSeries,
    inferSeriesName,
    sortFilesNaturally,
    buildTocGroups,
    groupForChapter,
    filterChapters,
    chapterNumberLabel,
    serviceToBookSettings,
    bookVoiceSettings,
    voiceDiffers,
    modelLabel,
    voiceSummary,
    buildChapterIndex,
    chapterCounts,
    mergeProgress,
    summarizeBook,
    chapterCountOf,
    classifyEpochChange,
    shouldSavePosition,
    withRetry,
    renderWindow,
    blockStatusLabel,
  };
  root.ReaderModel = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
