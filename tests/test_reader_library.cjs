// Library grouping, table-of-contents groups and per-book voice helpers.
const test = require('node:test');
const assert = require('node:assert/strict');
const Model = require('../web/novel_reader/reader-model.js');

test('files sort in reading order, including Chinese numerals and 上/中/下', () => {
  const names = (files) => Model.sortFilesNaturally(files.map((name) => ({ name }))).map((file) => file.name);
  assert.deepEqual(
    names(['凡人 第十卷.txt', '凡人 第二卷.txt', '凡人 第一卷.txt']),
    ['凡人 第一卷.txt', '凡人 第二卷.txt', '凡人 第十卷.txt'],
  );
  assert.deepEqual(names(['三体（下）.txt', '三体（上）.txt', '三体（中）.txt']), ['三体（上）.txt', '三体（中）.txt', '三体（下）.txt']);
  assert.deepEqual(names(['x-10.txt', 'x-2.txt']), ['x-2.txt', 'x-10.txt']);
});

test('series names are inferred from a shared file-name prefix', () => {
  assert.equal(Model.inferSeriesName(['斗破苍穹-01.txt', '斗破苍穹-12.txt']), '斗破苍穹');
  assert.equal(Model.inferSeriesName(['三体（上）.txt', '三体（下）.txt']), '三体');
  assert.equal(Model.inferSeriesName(['凡人修仙传 第一卷.txt', '凡人修仙传 第二卷.txt']), '凡人修仙传');
  assert.equal(Model.inferSeriesName(['a.txt', 'b.txt']), '');
  assert.equal(Model.inferSeriesName(['单本.md']), '单本');
});

test('library groups series together and sorts by recent activity', () => {
  const books = [
    { id: 1, name: '仙界篇', series: '凡人', created_at: 10 },
    { id: 2, name: '三体', created_at: 50 },
    { id: 3, name: '凡人修仙传', series: '凡人', created_at: 5, last_read_at: 100 },
  ];
  const entries = Model.groupLibrary(books, { sort: 'recent' });
  assert.deepEqual(entries.map((entry) => entry.series || entry.books[0].name), ['凡人', '三体']);
  assert.deepEqual(entries[0].books.map((book) => book.id), [3, 1]);
  const searched = Model.groupLibrary(books, { query: '三' });
  assert.deepEqual(searched.map((entry) => entry.books.map((book) => book.id)), [[2]]);
  assert.deepEqual(Model.knownSeries(books), ['凡人']);
});

test('table of contents groups by volume, or by 100 chapters for long flat books', () => {
  const volumes = [{ index: 0, title: '第一卷' }, { index: 1, title: '第二卷' }];
  const chapters = [
    { title: '序' },
    { title: 'a', volume_index: 0 },
    { title: 'b', volume_index: 0 },
    { title: 'c', volume_index: 1 },
  ];
  const groups = Model.buildTocGroups(chapters, volumes);
  assert.deepEqual(groups.map((group) => [group.title, group.start, group.end]), [
    ['未分卷', 0, 0], ['第一卷', 1, 2], ['第二卷', 3, 3],
  ]);
  assert.equal(Model.groupForChapter(groups, 2).title, '第一卷');

  const flat = Array.from({ length: 120 }, (_, i) => ({ title: `c${i}` }));
  assert.deepEqual(Model.buildTocGroups(flat, []).map((group) => group.title), [null]);
  const long = Array.from({ length: 250 }, (_, i) => ({ title: `c${i}` }));
  assert.deepEqual(
    Model.buildTocGroups(long, []).map((group) => [group.title, group.start, group.end]),
    [['第 1–100 章', 0, 99], ['第 101–200 章', 100, 199], ['第 201–250 章', 200, 249]],
  );
});

test('chapter search matches titles or a chapter number', () => {
  const chapters = [{ title: '第一章 雨夜' }, { title: '第二章 重逢' }, { title: '番外 雨后' }];
  assert.deepEqual(Model.filterChapters(chapters, '雨'), [0, 2]);
  assert.deepEqual(Model.filterChapters(chapters, '2'), [1]);
  assert.deepEqual(Model.filterChapters(chapters, ''), []);
});

test('book voice settings come from the book, not the service default', () => {
  const service = { model_profile: 'qwen_1_7b', reference_audio_path: '/svc.wav', qwen_seed: 42, qwen_aac_bitrate: '96k' };
  const saved = {
    model_profile: 'qwen_0_6b', reference_audio_path: '/book.wav', seed: 777321, configured_seed: -1,
    seed_mode: 'random', aac_bitrate: '64k', language: 'Chinese',
  };
  const settings = Model.bookVoiceSettings(saved, service);
  assert.equal(settings.reference_audio_path, '/book.wav');
  assert.equal(settings.model_profile, 'qwen_0_6b');
  assert.equal(settings.seed, -1);
  assert.equal(settings.aac_bitrate, '64k');
  assert.equal(settings.language, 'Chinese');
  assert.ok(Model.voiceDiffers(settings, Model.serviceToBookSettings(service)));
  assert.ok(!Model.voiceDiffers(settings, { ...settings, aac_bitrate: '128k' }));
  assert.equal(Model.voiceSummary(settings), '0.6B 极速 · 随机 Seed · AAC 64k');
});

test('summaries computed from a full book carry the reading chapter', () => {
  const summary = Model.summarizeBook({
    id: 'x',
    chapters: [{ index: 0, title: '一', segment_start: 0 }, { index: 1, title: '二', segment_start: 3 }],
    volumes: [{ index: 0 }],
    segments: [],
    playback: { segment_index: 4, captured_at: 2000 },
  });
  assert.equal(summary.reading_chapter_title, '二');
  assert.equal(summary.volume_count, 1);
  assert.equal(summary.last_read_at, 2);
  assert.equal(summary.chapters, undefined);
});
