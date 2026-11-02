const assert = require('node:assert/strict');
const { test } = require('node:test');
const Model = require('../web/novel_reader/reader-model.js');
const { loadReader, pcmBody, sampleBook, tick } = require('./reader_harness.cjs');

const json = (body, status = 200) => ({ ok: status < 400, status, json: async () => body, text: async () => JSON.stringify(body) });

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

// ---------------------------------------------------------------- model ----

test('chapter index groups segments linearly and maps positions', () => {
  const book = sampleBook(6, [
    { index: 0, title: 'a', segment_start: 0, segment_end: 1 },
    { index: 1, title: 'b', segment_start: 2, segment_end: 5 },
  ]);
  const index = Model.buildChapterIndex(book);
  assert.deepEqual(index.byChapter.map((list) => list.map((s) => s.index)), [[0, 1], [2, 3, 4, 5]]);
  assert.equal(index.chapterOf.get(4), 1);
  assert.equal(index.position.get(3), 3);
});

test('progress merge patches changed segments, keeps text, and flags text revisions', () => {
  const book = sampleBook(3);
  const index = Model.buildChapterIndex(book);
  const result = Model.mergeProgress(book, {
    id: 'book', revision: 4, text_revision: 1, state: 'running', stats: { progress: 0.33 },
    segments: [{ index: 1, status: 'completed', audio_file: 'a.m4a', duration_seconds: 3, rev: 4 },
      { index: 2, status: 'pending', rev: 1 }],
  }, index);
  assert.deepEqual(result.changed, [1]);
  assert.equal(result.needsFullReload, false);
  assert.equal(book.segments[1].text, '第1块');
  assert.equal(book.segments[1].status, 'completed');
  assert.equal(book.revision, 4);
  assert.equal(book.state, 'running');
  assert.equal(Model.mergeProgress(book, { id: 'book', text_revision: 2, segments: [] }, index).needsFullReload, true);
  assert.equal(Model.mergeProgress(book, { id: 'other', segments: [] }, index).ignored, true);
  assert.equal(Model.mergeProgress(book, { id: 'book', text_revision: 1, segments: [{ index: 99 }] }, index).needsFullReload, true);
});

test('epoch changes distinguish a global stop from a service restart', () => {
  assert.equal(Model.classifyEpochChange(1000, 1000), 'none');
  assert.equal(Model.classifyEpochChange(1760000000000, 1760000000001), 'stop');
  assert.equal(Model.classifyEpochChange(1760000000000, 1760000090000), 'restart');
  assert.equal(Model.classifyEpochChange(1760000000005, 1760000000000), 'restart');
});

test('position saves are throttled and skip unchanged positions', () => {
  const last = { index: 2, offset: 10, at: 1000 };
  assert.equal(Model.shouldSavePosition(null, { index: 2, offset: 0 }, 0), true);
  assert.equal(Model.shouldSavePosition(last, { index: 2, offset: 10.3 }, 99999), false, 'not moved');
  assert.equal(Model.shouldSavePosition(last, { index: 2, offset: 13 }, 3000), false, 'too soon');
  assert.equal(Model.shouldSavePosition(last, { index: 2, offset: 13 }, 7000), true);
  assert.equal(Model.shouldSavePosition(last, { index: 3, offset: 0 }, 1001), true, 'new block saves at once');
  assert.equal(Model.shouldSavePosition(last, { index: 2, offset: 11 }, 1001, { force: true }), true);
});

test('retry helper retries transient failures only', async () => {
  let attempts = 0;
  const value = await Model.withRetry(async () => { attempts += 1; if (attempts < 3) throw new Error('x'); return 7; },
    { sleep: async () => {} });
  assert.equal(value, 7);
  attempts = 0;
  await assert.rejects(Model.withRetry(async () => { attempts += 1; const e = new Error('stale'); e.status = 409; throw e; },
    { sleep: async () => {}, shouldRetry: (e) => e.status !== 409 }), /stale/);
  assert.equal(attempts, 1);
});

// ------------------------------------------------------------- reader -----

test('ticker saves position only while audio is playing and moving', async () => {
  const app = loadReader();
  const saves = [];
  app.run(`fetch = async (url) => { if (String(url).endsWith('/playback')) globalThis.__saves.push(url); return {ok:true,json:async()=>({ok:true})}; };`);
  app.context.__saves = saves;
  app.context.setBook(sampleBook(3));
  Object.assign(app.state, { player: 'playing', playbackMode: 'quality', playingBlock: 1 });
  app.state.qualityAudio.src = 'blob:x';
  app.state.qualityAudio.currentTime = 10;
  app.context.maybeSavePosition();
  assert.equal(saves.length, 1);
  app.context.maybeSavePosition();
  app.state.qualityAudio.currentTime = 10.2;
  app.context.maybeSavePosition();
  app.state.qualityAudio.currentTime = 13;
  app.context.maybeSavePosition();
  assert.equal(saves.length, 1, 'unchanged or too-soon positions are not sent');
  app.state.lastSavedPosition.at -= 6000;
  app.context.maybeSavePosition();
  assert.equal(saves.length, 2);
  app.state.qualityAudio.currentTime = 30;
  app.state.lastSavedPosition.at -= 6000;
  app.state.player = 'paused';
  app.context.maybeSavePosition();
  assert.equal(saves.length, 2, 'paused audio is not saved by the ticker');
  app.state.player = 'playing';
  app.context.flushPosition();
  assert.equal(saves.length, 3, 'explicit flush (pause/stop) saves a moved position');
});

test('opening another book resets generation context and ignores stale opens', async () => {
  const app = loadReader();
  const bookA = { ...sampleBook(2), id: 'A', state: 'running' };
  const bookB = { ...sampleBook(3), id: 'B', state: 'ready' };
  const slowA = deferred();
  app.context.setBook(bookA);
  app.state.qualityContext = { bookId: 'A', title: 'A' };
  app.state.playbackMode = 'quality-generating';
  app.context.jsonFetch = async (path) => (String(path).endsWith('/A') ? slowA.promise : bookB);
  const first = app.context.openBook('A');
  await app.context.openBook('B');
  slowA.resolve(bookA);
  await first;
  assert.equal(app.state.book.id, 'B');
  assert.equal(app.state.qualityContext, null);
  assert.equal(app.state.playbackMode, '');
  assert.equal(app.elements.get('start-listening').disabled, false);
  assert.equal(app.elements.get('generate-whole-book').disabled, false);
});

test('a progress poll for the previous book never touches the newly opened one', async () => {
  const app = loadReader();
  const toasts = [];
  app.context.toast = (message) => toasts.push(message);
  const bookA = { ...sampleBook(2), id: 'A', state: 'running' };
  const bookB = { ...sampleBook(2), id: 'B', state: 'ready' };
  const progressA = deferred();
  app.context.setBook(bookA);
  app.state.qualityContext = { bookId: 'A', title: 'A' };
  app.context.jsonFetch = async (path) => (String(path).includes('/progress') ? progressA.promise : bookB);
  const poll = app.context.pollBookProgress();
  await app.context.openBook('B');
  progressA.resolve({ id: 'A', revision: 9, text_revision: 1, state: 'completed', final_audio: 'x.m4a', segments: [] });
  await poll;
  assert.equal(app.state.book.id, 'B');
  assert.equal(app.state.book.state, 'ready');
  assert.deepEqual(toasts, []);
  clearTimeout(app.state.pollTimer);
});

test('progress updates patch only changed block elements and chapter counts', () => {
  const app = loadReader();
  const book = sampleBook(4, [
    { index: 0, title: 'a', segment_start: 0, segment_end: 1 },
    { index: 1, title: 'b', segment_start: 2, segment_end: 3 },
  ]);
  app.context.setBook(book);
  app.state.chapterIndex = 0;
  app.context.renderBook();
  const block0 = app.state.blockElements.get(0);
  const block1 = app.state.blockElements.get(1);
  const reload = app.context.applyProgress({ id: 'book', revision: 2, text_revision: 1, state: 'running',
    segments: [{ index: 1, status: 'generating', rev: 2 }] });
  assert.equal(reload, false);
  assert.equal(app.state.blockElements.get(1), block1, 'element is patched, not rebuilt');
  assert.equal(block1.classList.contains('generating'), true);
  assert.equal(block0.classList.contains('generating'), false);
  app.context.applyProgress({ id: 'book', revision: 3, text_revision: 1,
    segments: [{ index: 1, status: 'completed', duration_seconds: 4, audio_file: 'x', rev: 3 }] });
  assert.match(app.state.chapterButtons[0].querySelector('small').textContent, /1\/2 AAC/);
  assert.equal(app.context.applyProgress({ id: 'book', text_revision: 5, segments: [] }), true);
});

test('large chapters render a window around the current block', () => {
  const app = loadReader();
  app.context.setBook(sampleBook(1000));
  app.state.selectedBlock = 0;
  app.context.renderBook();
  const rendered = app.state.blockElements.size;
  assert.ok(rendered < 400, `rendered ${rendered}`);
  app.context.setPlayingBlock(900, { scroll: false });
  assert.ok(app.state.blockElements.has(900));
  assert.ok(app.state.blockElements.get(900).classList.contains('playing'));
  const tabbable = [...app.state.blockElements.values()].filter((block) => block.tabIndex === 0);
  assert.equal(tabbable.length, 1, 'roving tabindex');
});

test('segment text keeps line breaks for pre-line rendering', () => {
  const app = loadReader();
  const book = sampleBook(1);
  book.segments[0].text = '第一段\n第二段';
  app.context.setBook(book);
  app.context.renderBook();
  assert.equal(app.state.blockElements.get(0).querySelector('block-text').textContent, '第一段\n第二段');
});

test('pausing between blocks is respected: the next block never resumes audio', async () => {
  const app = loadReader({ contextState: 'suspended' });
  app.run(`fetch = async () => ({ok:true, body:{getReader:()=>({read:async()=>({done:true})})}, headers:{get:()=>null}});
    jsonFetch = async () => ({state:'finished'});`);
  app.state.paused = true;
  await app.context.receiveStream({ start: { job_id: 'j', sample_rate: 4, channels: 1 }, segmentIndex: 0 }, app.state.streamRunId);
  const context = app.contexts[0];
  assert.equal(context.resumeCalls, 0);
  Object.assign(app.state, { player: 'paused', playbackMode: 'stream' });
  app.context.togglePlayback();
  assert.equal(context.resumeCalls, 1);
  assert.equal(app.state.player, 'playing');
  // quality-online: the run loop waits while paused
  app.state.paused = true;
  let released = false;
  const wait = app.context.waitWhilePaused(app.state.streamRunId).then(() => { released = true; });
  await tick(200);
  assert.equal(released, false);
  app.state.paused = false;
  await wait;
  assert.equal(released, true);
});

test('stale callbacks from an earlier run cannot clobber a new run', async () => {
  const app = loadReader();
  const book = sampleBook(3);
  book.segments.forEach((segment) => Object.assign(segment, { status: 'completed', audio_file: `seg${segment.index}.m4a` }));
  app.context.setBook(book);
  app.context.playQualitySegments(app.context.completedListeningSegments(0));
  const oldAudio = app.state.qualityAudio;
  const staleEnded = oldAudio.onended;
  const staleError = oldAudio.onerror;
  assert.equal(app.state.playbackMode, 'quality');
  const runId = app.context.beginRun('stream');
  staleEnded();
  staleError();
  assert.equal(app.state.playbackMode, 'stream');
  assert.equal(app.state.player, 'starting');
  assert.equal(app.state.streamRunId, runId);
  // a stale stream reader is aborted without scheduling audio
  await assert.rejects(app.context.receiveStream({ start: { job_id: 'x', sample_rate: 4, channels: 1 } }, runId - 1),
    (error) => error.name === 'AbortError');
  assert.deepEqual(app.scheduled, []);
  app.context.stopLocal();
});

test('saved AAC playback swaps to the preloaded element on ended (gapless)', async () => {
  const app = loadReader();
  const book = sampleBook(3);
  book.segments.forEach((segment) => Object.assign(segment, { status: 'completed', audio_file: `seg${segment.index}.m4a` }));
  app.context.setBook(book);
  app.context.playQualitySegments(app.context.completedListeningSegments(0));
  await tick();
  const first = app.state.qualityAudio;
  const preload = app.state.preloadAudio;
  const primedPlays = preload.playCalls;
  first.oncanplaythrough();
  assert.match(preload.src, /seg1\.m4a/);
  first.onended();
  await tick();
  assert.equal(app.state.qualityAudio, preload, 'preloaded element became current');
  assert.equal(preload.playCalls, primedPlays + 1);
  assert.equal(app.state.playingBlock, 1);
  assert.match(app.state.preloadAudio.src || '', /^$|seg2/);
  app.context.stopLocal();
});

test('play/next clicks while starting are ignored; next crosses chapters', async () => {
  const app = loadReader();
  const book = sampleBook(4, [
    { index: 0, title: 'a', segment_start: 0, segment_end: 1 },
    { index: 1, title: 'b', segment_start: 2, segment_end: 3 },
  ]);
  app.context.setBook(book);
  app.context.renderBook();
  app.state.player = 'starting';
  app.state.selectedBlock = 1;
  const before = app.state.streamRunId;
  app.context.togglePlayback();
  app.context.moveBlock(1);
  assert.equal(app.state.streamRunId, before, 'no restart while starting');
  app.state.player = 'idle';
  app.context.moveBlock(1);
  assert.equal(app.state.selectedBlock, 2);
  assert.equal(app.state.chapterIndex, 1, 'moved into the next chapter');
});

test('streaming prefetches the next block and schedules it gaplessly', async () => {
  const app = loadReader({ autoClock: false });
  app.context.setBook(sampleBook(2));
  const calls = [];
  let jobs = 0;
  app.context.reserveContinuousListening = async () => {};
  app.context.fetch = async (url, options = {}) => {
    calls.push(`${options.method || 'GET'} ${url}`);
    if (url === '/api/generate-stream/start') {
      return json({ job_id: `job${jobs++}`, seed: 7, sample_rate: 4, channels: 1, playback_epoch: 5 });
    }
    if (url.endsWith('/audio')) return { ok: true, body: pcmBody([new Uint8Array(16)]), headers: { get: () => null } };
    if (url.endsWith('/status')) return json({ state: 'finished' });
    return json({ ok: true });
  };
  const run = app.context.startStreamChapter(0);
  try {
  await tick(60);
  const context = app.contexts[0];
  assert.equal(context.time, 0, 'block 0 has not finished playing yet');
  const relevant = calls.filter((call) => !call.includes('/playback'));
  assert.deepEqual(relevant, [
    'POST /api/generate-stream/start',
    'GET /api/generate-stream/job0/audio',
    'GET /api/generate-stream/job0/status',
    'POST /api/generate-stream/start',
    'GET /api/generate-stream/job1/audio',
    'POST /api/generate-stream/job0/close',
    'GET /api/generate-stream/job1/status',
  ]);
  assert.equal(app.sources.length, 2);
  assert.equal(app.sources[1].startedAt, app.sources[0].startedAt + 2, 'next block starts exactly when block 0 ends');
  assert.deepEqual([...app.state.streamTimeline.map((entry) => entry.segmentIndex)], [0, 1]);
  const form = app.context.streamForm(app.state.book.segments[0]);
  assert.equal(form.get('language'), 'auto');
  assert.equal(form.get('max_new_tokens'), null);
  context.time = 1;
  app.context.playerTick();
  assert.equal(app.state.playingBlock, 0);
  context.time = 2.5;
  app.context.playerTick();
  assert.equal(app.state.playingBlock, 1, 'UI follows the audible block');
  context.time = 10;
  await run;
  assert.equal(app.elements.get('playback-mode').textContent, '本书播放完成');
  assert.ok(calls.includes('POST /api/generate-stream/job1/close'));
  assert.equal(app.state.player, 'idle');
  } finally {
    app.context.stopLocal();
  }
});

test('streaming locks the rate control to 1x with a hint', () => {
  const app = loadReader();
  app.context.setBook(sampleBook(1));
  app.context.beginRun('stream');
  assert.equal(app.elements.get('playback-rate').disabled, true);
  assert.equal(app.elements.get('rate-hint').classList.contains('hidden'), false);
  app.context.stopLocal();
  assert.equal(app.elements.get('playback-rate').disabled, false);
});

test('deleting a book clears its localStorage keys', async () => {
  const app = loadReader();
  const store = app.context.localStorage;
  store.setItem('qwen-reader-position-B', '{}');
  store.setItem('qwen-reader-chapter-B', '2');
  store.setItem('qwen-reader-book', 'B');
  app.context.forgetBookStorage('B');
  assert.equal(store.getItem('qwen-reader-position-B'), null);
  assert.equal(store.getItem('qwen-reader-chapter-B'), null);
  assert.equal(store.getItem('qwen-reader-book'), null);
});

test('stop button ends playback, keeps the position, and is disabled when idle', () => {
  const app = loadReader();
  const book = sampleBook(3);
  book.segments.forEach((segment) => Object.assign(segment, { status: 'completed', audio_file: `seg${segment.index}.m4a` }));
  app.context.setBook(book);
  const stop = app.context.document.getElementById('stop-playback');
  app.context.setActionsEnabled(true);
  assert.equal(stop.disabled, true, 'nothing to stop while idle');
  app.context.playQualitySegments(app.context.completedListeningSegments(0));
  assert.notEqual(app.state.player, 'idle');
  assert.equal(stop.disabled, false);
  const runId = app.state.streamRunId;
  app.context.stopListening();
  assert.equal(app.state.player, 'idle');
  assert.equal(app.state.stopped, true);
  assert.notEqual(app.state.streamRunId, runId, 'stale callbacks of the stopped run are invalidated');
  assert.equal(stop.disabled, true);
  assert.equal(app.context.document.getElementById('playback-mode').textContent, '已停止');
});

test('play starts from saved AAC instead of re-synthesising generated blocks', async () => {
  const app = loadReader();
  const book = sampleBook(3);
  book.segments.slice(0, 2).forEach((segment) => Object.assign(segment, { status: 'completed', audio_file: `seg${segment.index}.m4a` }));
  app.context.setBook(book);
  let liveStarts = 0;
  app.context.startStreamChapter = () => { liveStarts += 1; };
  app.context.startQualityChapter = () => { liveStarts += 1; };
  app.state.selectedBlock = 0;
  app.context.startSelectedListening();
  await tick();
  assert.equal(app.state.playbackMode, 'quality');
  assert.match(app.state.qualityAudio.src, /seg0\.m4a/);
  assert.equal(liveStarts, 0);
  app.context.stopLocal();
});

test('when saved audio runs out playback continues live from the next block', async () => {
  const app = loadReader();
  const book = sampleBook(3);
  Object.assign(book.segments[0], { status: 'completed', audio_file: 'seg0.m4a' });
  app.context.setBook(book);
  const live = [];
  app.context.startStreamChapter = (index) => { live.push(index); };
  app.context.playQualitySegments(app.context.completedListeningSegments(0));
  await tick();
  app.state.qualityAudio.onended();
  await tick();
  assert.deepEqual(live, [1]);
});

test('during whole-book generation finished blocks stay playable and new ones join the queue', async () => {
  const app = loadReader();
  const book = sampleBook(3);
  book.state = 'running';
  Object.assign(book.segments[0], { status: 'completed', audio_file: 'seg0.m4a' });
  app.context.setBook(book);
  let liveStarts = 0;
  app.context.startStreamChapter = () => { liveStarts += 1; };
  app.state.selectedBlock = 0;
  app.context.startSelectedListening();
  await tick();
  assert.equal(app.state.playbackMode, 'quality');
  // Block 1 finishes while block 0 is playing.
  Object.assign(book.segments[1], { status: 'completed', audio_file: 'seg1.m4a' });
  app.state.qualityAudio.onended();
  await tick();
  assert.equal(app.state.playingBlock, 1);
  app.state.qualityAudio.onended();
  await tick();
  assert.equal(liveStarts, 0, 'never synthesises live while the book is generating');
  assert.equal(app.state.player, 'idle');
});

test('generation range covers the current chapter, volume, rest or whole book', () => {
  const app = loadReader();
  const book = sampleBook(6, [
    { index: 0, title: 'a', segment_start: 0, segment_end: 1, volume_index: 0 },
    { index: 1, title: 'b', segment_start: 2, segment_end: 3, volume_index: 0 },
    { index: 2, title: 'c', segment_start: 4, segment_end: 5, volume_index: 1 },
  ]);
  book.volumes = [{ index: 0, title: '卷一' }, { index: 1, title: '卷二' }];
  app.context.setBook(book);
  app.state.chapterIndex = 1;
  const range = (scope) => { const r = app.context.generationRange(scope); return [r.start, r.end]; };
  assert.deepEqual(range('chapter'), [2, 3]);
  assert.deepEqual(range('volume'), [0, 3]);
  assert.deepEqual(range('rest'), [2, 5]);
  assert.deepEqual(range('book'), [0, 5]);
});
