const assert = require('node:assert/strict');
const { test } = require('node:test');
const { loadReader, sampleBook } = require('./reader_harness.cjs');

const reader = () => loadReader();

for (const state of ['error', 'interrupted', 'running']) test(`stream ending in ${state} fails`, async () => {
  const app = reader();
  app.run(`fetch = async () => ({ok:true, body:{getReader:()=>({read:async()=>({done:true})})}, headers:{get:()=>null}});
    jsonFetch = async () => ({state: '${state}', error:'模型失败'});`);
  await assert.rejects(app.playStreamingAudio({job_id:'test',sample_rate:4,channels:1}), /模型失败/);
});

test('PCM resume skips the saved audio and handles split sample bytes', async () => {
  const app = reader();
  app.state.playingBlock = 3;
  app.state.resumePosition = { segment_index: 3, offset_seconds: 1 };
  app.run(`let chunks = [new Uint8Array(1), new Uint8Array(15)];
    fetch = async () => ({ok:true, body:{getReader:()=>({read:async()=>chunks.length?{done:false,value:chunks.shift()}:{done:true}})}, headers:{get:()=>null}});
    jsonFetch = async () => ({state:'finished'});`);
  await app.playStreamingAudio({job_id:'test',sample_rate:4,channels:1});
  assert.deepEqual(app.scheduled, [4]);
});

test('truncated PCM sample is rejected', async () => {
  const app = reader();
  app.run(`let chunk = new Uint8Array(1);
    fetch = async () => ({ok:true,body:{getReader:()=>({read:async()=>chunk?{done:false,value:((x)=>{chunk=null;return x;})(chunk)}:{done:true}})},headers:{get:()=>null}});
    jsonFetch = async () => ({state:'finished'});`);
  await assert.rejects(app.playStreamingAudio({job_id:'test',sample_rate:4,channels:1}), /音频流不完整/);
});

test('stream position follows actual playback time and rate', () => {
  const app = reader();
  Object.assign(app.state, {book:{id:'book'}, playingBlock:2, playbackMode:'stream',
    audioContext:{state:'running', currentTime:12},streamOffset:4,
    streamTimeline:[{start:10,end:14,audioStart:4,audioEnd:12,rate:2}]});
  assert.equal(app.currentPlaybackPosition().offset, 8);
  app.state.audioContext.currentTime = 14;
  assert.equal(app.currentPlaybackPosition().offset, 12);
});

test('AAC resume seeks after metadata and clamps to duration', () => {
  const app = reader();
  app.state.resumePosition = {segment_index:5,offset_seconds:7.25};
  const audio = {duration:10,currentTime:0};
  app.seekSavedPosition(audio,5); audio.onloadedmetadata();
  assert.equal(audio.currentTime,7.25);
  app.seekSavedPosition(audio,4); audio.currentTime=0; audio.onloadedmetadata();
  assert.equal(audio.currentTime,0);
});

test('failed block does not advance the book or report completion', async () => {
  const app = reader();
  app.context.setBook(sampleBook(2));
  app.run(`let attempted=[]; let saved=[];
    reserveContinuousListening=async()=>{};
    startStreamJob=async(segment)=>{attempted.push(segment.index);throw new Error('故障');};
    savePlayback=async(index)=>{saved.push(index);}; toast=()=>{};
    globalThis.results={attempted,saved};`);
  await app.startStreamChapter(0);
  assert.equal(String(app.context.results.attempted),'0');
  assert.equal(app.context.results.saved.length,0);
  assert.match(app.elements.get('playback-mode').textContent,/播放失败/);
  assert.equal(app.state.player, 'idle');
});

test('book refresh updates library progress as well as the open chapter', async () => {
  const app=reader();
  app.context.setBook({id:'book',state:'running'});
  app.state.books=[{id:'book',state:'running',stats:{progress:0}}];
  app.run(`jsonFetch=async()=>({id:'book',state:'completed',stats:{progress:1}});`);
  await app.refreshCurrentBook(false);
  assert.equal(app.state.books[0].stats.progress,1);
  assert.equal(app.state.books[0].segments, undefined, 'library keeps summaries only');
  assert.equal(app.state.book.state,'completed');
});
