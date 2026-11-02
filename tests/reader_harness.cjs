// Shared vm harness for the novel reader scripts (reader-model.js + reader.js)
// with a tiny fake DOM, fake Audio and fake Web Audio.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ROOT = path.join(__dirname, '..', 'web', 'novel_reader');

class FakeClassList {
  constructor(owner) { this.owner = owner; this.set = new Set(); }
  add(...names) { names.forEach((name) => this.set.add(name)); }
  remove(...names) { names.forEach((name) => this.set.delete(name)); }
  toggle(name, force) {
    const on = force === undefined ? !this.set.has(name) : Boolean(force);
    if (on) this.set.add(name); else this.set.delete(name);
    return on;
  }
  contains(name) { return this.set.has(name); }
}

function makeDom() {
  const doc = { activeElement: null, hidden: false };
  class FakeElement {
    constructor(tag = 'div', id = '') {
      this.tagName = String(tag).toUpperCase();
      this.id = id;
      this.children = [];
      this.classList = new FakeClassList(this);
      this.dataset = {};
      this.style = { setProperty() {} };
      this.attributes = {};
      this.textContent = '';
      this.value = '1';
      this.disabled = false;
      this.listeners = {};
      this.parentNode = null;
      this.tabIndex = -1;
    }
    set className(value) {
      this.classList = new FakeClassList(this);
      String(value).split(/\s+/).filter(Boolean).forEach((name) => this.classList.add(name));
    }
    get className() { return [...this.classList.set].join(' '); }
    set innerHTML(value) { this._html = value; this.children = []; }
    get innerHTML() { return this._html || ''; }
    get lastChild() {
      if (this.children.length) return this.children[this.children.length - 1];
      if (!this._text) this._text = { textContent: '' };
      return this._text;
    }
    get nextElementSibling() {
      const list = this.parentNode?.children || [];
      return list[list.indexOf(this) + 1] || null;
    }
    get previousElementSibling() {
      const list = this.parentNode?.children || [];
      return list[list.indexOf(this) - 1] || null;
    }
    appendChild(child) {
      if (child.isFragment) {
        const moved = child.children;
        child.children = [];
        moved.forEach((item) => this.appendChild(item));
        return child;
      }
      child.parentNode = this;
      this.children.push(child);
      return child;
    }
    append(...children) { children.forEach((child) => this.appendChild(child)); }
    replaceChildren(...children) { this.children = []; children.forEach((child) => this.appendChild(child)); }
    remove() {
      if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((item) => item !== this);
      this.parentNode = null;
    }
    setAttribute(key, value) { this.attributes[key] = String(value); }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; }
    addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
    querySelectorAll(selector) {
      const classes = selector.split('.').filter(Boolean);
      const out = [];
      const walk = (node) => {
        for (const child of node.children || []) {
          if (classes.every((name) => child.classList?.contains(name))) out.push(child);
          walk(child);
        }
      };
      walk(this);
      return out;
    }
    querySelector(selector) {
      const found = this.querySelectorAll(selector)[0];
      if (found) return found;
      // innerHTML is not parsed; hand out a stable stand-in for markup we set.
      const name = selector.split('.').filter(Boolean).pop();
      if (this._html && name && this._html.includes(name)) {
        this._virtual ||= new Map();
        if (!this._virtual.has(name)) {
          const child = new FakeElement('span');
          child.className = name;
          child.parentNode = this;
          this._virtual.set(name, child);
        }
        return this._virtual.get(name);
      }
      return null;
    }
    closest(selector) {
      const name = selector.replace(/^\./, '');
      let node = this;
      while (node) {
        if (node.classList?.contains(name)) return node;
        node = node.parentNode;
      }
      return null;
    }
    focus() { doc.activeElement = this; }
    blur() { if (doc.activeElement === this) doc.activeElement = null; }
    scrollIntoView() {}
    scrollTo() {}
    showModal() { this.open = true; }
    close() { this.open = false; }
    click() { this.onclick?.(); }
  }
  const elements = new Map();
  Object.assign(doc, {
    body: new FakeElement('body'),
    documentElement: new FakeElement('html'),
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new FakeElement('div', id));
      return elements.get(id);
    },
    createElement(tag) { return new FakeElement(tag); },
    createDocumentFragment() { const fragment = new FakeElement('#fragment'); fragment.isFragment = true; return fragment; },
    querySelector() { return null; },
    addEventListener() {},
  });
  return { doc, elements, FakeElement };
}

// options.autoClock: currentTime advances by 10 on every read (legacy tests).
function loadReader(options = {}) {
  const { doc, elements, FakeElement } = makeDom();
  const scheduled = [];
  const sources = [];
  const contexts = [];
  class AudioContext {
    constructor() {
      this.state = options.contextState || 'running';
      this.time = 0;
      this.resumeCalls = 0;
      this.suspendCalls = 0;
      this.destination = {};
      contexts.push(this);
    }
    get currentTime() { return options.autoClock === false ? this.time : (this.time += 10); }
    set currentTime(value) { this.time = value; }
    async resume() { this.resumeCalls += 1; this.state = 'running'; }
    async suspend() { this.suspendCalls += 1; this.state = 'suspended'; }
    async close() { this.state = 'closed'; }
    createBuffer(channels, frames, sampleRate) {
      scheduled.push(frames);
      return { duration: frames / sampleRate, getChannelData: () => new Float32Array(frames) };
    }
    createBufferSource() {
      const source = { playbackRate: {}, connect() {}, disconnect() {}, start(at) { source.startedAt = at; },
        stop() { source.stopped = true; } };
      sources.push(source);
      return source;
    }
  }
  class FakeAudio extends FakeElement {
    constructor() { super('audio'); this.src = ''; this.currentTime = 0; this.paused = true; this.playCalls = 0; }
    play() { this.playCalls += 1; this.paused = false; return Promise.resolve(); }
    pause() { this.paused = true; }
    load() {}
    removeAttribute(key) { if (key === 'src') this.src = ''; super.removeAttribute(key); }
  }
  const timers = { setTimeout, clearTimeout, setInterval, clearInterval };
  const context = vm.createContext({
    console, FormData, AbortController, AbortSignal, DOMException, URL, Blob, Promise,
    ...timers,
    Audio: FakeAudio,
    localStorage: (() => {
      const store = new Map();
      return { setItem: (k, v) => store.set(k, String(v)), getItem: (k) => (store.has(k) ? store.get(k) : null),
        removeItem: (k) => store.delete(k), store };
    })(),
    navigator: { sendBeacon: () => true },
    window: { AudioContext, ...timers, location: { pathname: '/reader', search: '', assign() {} }, addEventListener() {} },
    document: doc,
    fetch: async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => '' }),
  });
  const source = fs.readFileSync(path.join(ROOT, 'reader-model.js'), 'utf8')
    + '\n' + fs.readFileSync(path.join(ROOT, 'reader.js'), 'utf8').replace(/\ninit\(\);\s*$/, '')
    + '\nglobalThis.production = {state, PLAYER, Model};';
  vm.runInContext(source, context, { filename: 'reader-bundle.js' });
  const run = (code) => vm.runInContext(code, context);
  return { context, run, state: context.production.state, PLAYER: context.production.PLAYER,
    scheduled, sources, contexts, elements, doc, ...context.production, ...pick(context) };
}

function pick(context) {
  const names = ['playStreamingAudio', 'currentPlaybackPosition', 'seekSavedPosition', 'startStreamChapter',
    'streamForm', 'refreshCurrentBook', 'receiveStream'];
  return Object.fromEntries(names.map((name) => [name, context[name]]));
}

function pcmBody(chunks) {
  const queue = chunks.slice();
  return { getReader: () => ({ read: async () => (queue.length ? { done: false, value: queue.shift() } : { done: true }) }) };
}

function sampleBook(segmentCount = 4, chapters = null) {
  const segments = Array.from({ length: segmentCount }, (_, index) => ({
    index, text: `第${index}块`, status: 'pending', rev: 1,
  }));
  return {
    id: 'book', name: '书', revision: 1, text_revision: 1, state: 'ready', stats: { progress: 0 },
    segments,
    chapters: chapters || [{ index: 0, title: '一', segment_start: 0, segment_end: segmentCount - 1 }],
  };
}

const tick = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));

module.exports = { loadReader, pcmBody, sampleBook, tick };
