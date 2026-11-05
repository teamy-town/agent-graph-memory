/**
 * Built-in embeddings: configured by the caller, computed on `put()`,
 * rebuilt by `reembed()`, searched by `recall({ locate })` and
 * `match({ semantic })`.
 *
 * The caller's `EmbeddingConfig` decides WHAT is embedded (`text`, `kinds`,
 * `maxChars`), because embedding every node of a busy graph is mostly waste:
 * runs and file paths are found by id, and only prose-bearing kinds reward a
 * vector. This file only knows how to turn the decision into rows.
 *
 * Storage is one row per node in `node_vectors`, holding the vector of the
 * version it was computed from. SQLite keeps a BLOB and cosine runs here, in
 * JS, over every stored vector of the requested kinds — brute force, which is
 * fine at the sizes an agent memory reaches before anyone needs an index.
 * Postgres with pgvector keeps a `vector` and orders by `<=>` (cosine
 * distance) in SQL; without pgvector it keeps BYTEA and takes the JS path.
 *
 * A BLOB is written as int8 — one byte per dimension plus a float32 scale
 * (`encodeInt8`), a quarter of float32. Ranking is what a vector is for, and
 * measured on a real 15.6k-vector fleet memory (OpenAI text-embedding-3-small,
 * 1536 dims, 200 queries) the int8 top-10 agrees with the float32 top-10 on
 * 99.6% of results (top-1: 99%); cutting dimensions to 512 instead agreed on
 * 73%. Blobs written earlier as float32 are still read (told apart by length:
 * `dims * 4` against `dims + 4` bytes); a `reembed()` rewrites them. The
 * search reads each blob through a typed-array view — no per-element copy —
 * and streams the rows into a top-k list, so a query holds k candidates, not
 * every vector of the graph.
 *
 * A failed embedding never fails a write: the node is committed first, and a
 * model that is down loses a vector, not a fact. The failure is logged.
 */

import type { SqlParam, SqlValue } from './driver.js';
import type { HandleContext } from './graph.js';
import { SQL, rowToNode, num, type SqlRow } from './schema.js';
import type { CompactVectorsResult, EmbeddingConfig, NodeId, NodeRecord, ReembedOptions, ReembedResult } from './types.js';

export const DEFAULT_MAX_CHARS = 2000;
/** Texts per model call in `reembed()`. */
const BATCH = 64;

// ---------------------------------------------------------------------------
// The rule: which nodes, which text

/** The text the rule selects for a node, or null when the node is not to be
 *  embedded (kind not listed, or `text` declined). */
export function embeddingText(cfg: EmbeddingConfig, node: NodeRecord): string | null {
  if (cfg.kinds && !cfg.kinds.includes(node.kind)) return null;
  const raw = cfg.text ? cfg.text(node) : node.label;
  if (raw === null || raw === undefined) return null;
  const text = String(raw).trim();
  if (text.length === 0) return null;
  const max = cfg.maxChars ?? DEFAULT_MAX_CHARS;
  return text.length > max ? text.slice(0, max) : text;
}

/** How the HTTP server (and a config file) name a text rule without code. */
export type TextRuleSpec = 'label' | 'label+attrs' | { attrs: string[] };

/** A `text` function from a declarative spec: the label alone; the label
 *  plus every attr as `key: value` lines; or the label plus the listed attrs.
 *  Objects are JSON, so a nested value still reads as text. */
export function textRuleFrom(spec: TextRuleSpec | undefined): (node: NodeRecord) => string | null {
  if (spec === undefined || spec === 'label') return (n) => n.label;
  const pick = spec === 'label+attrs' ? undefined : new Set(spec.attrs);
  return (n) => {
    const lines = [n.label];
    for (const [k, v] of Object.entries(n.attrs)) {
      if (pick && !pick.has(k)) continue;
      if (v === null || v === undefined) continue;
      lines.push(`${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`);
    }
    return lines.join('\n');
  };
}

// ---------------------------------------------------------------------------
// Vectors as stored

/** float32 little-endian — how a BLOB was written before int8 (still read). */
export function encodeFloat32(v: number[]): Uint8Array {
  const f = new Float32Array(v);
  return new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
}

/** int8, symmetric per vector: `dims` signed bytes (each value / scale,
 *  rounded) followed by the scale as float32 little-endian, where scale =
 *  max |value| / 127. Cosine is scale-invariant, so the search never needs
 *  the scale; it is kept so `decodeVector` gives the values back. */
export function encodeInt8(v: number[]): Uint8Array {
  let max = 0;
  for (const x of v) max = Math.max(max, Math.abs(x));
  const scale = max / 127;
  const out = new Uint8Array(v.length + 4);
  const q = new Int8Array(out.buffer, 0, v.length);
  if (scale > 0) for (let i = 0; i < v.length; i++) q[i] = Math.round(v[i] / scale);
  new DataView(out.buffer).setFloat32(v.length, scale, true);
  return out;
}

/** A stored BLOB as a typed array over its own bytes — no per-element copy.
 *  `dims + 4` bytes is int8 (`encodeInt8`); `dims * 4` is the older float32. */
function blobView(raw: Uint8Array, dims: number): Int8Array | Float32Array {
  if (raw.byteLength === dims + 4) return new Int8Array(raw.buffer, raw.byteOffset, dims);
  if (raw.byteLength === dims * 4) {
    const bytes = raw.byteOffset % 4 === 0 ? raw : new Uint8Array(raw); // Float32Array needs 4-byte alignment
    return new Float32Array(bytes.buffer, bytes.byteOffset, dims);
  }
  throw new Error(`agent-graph: a stored vector of ${raw.byteLength} bytes is neither int8 nor float32 of ${dims} dims`);
}

/** Whatever the engine handed back — bytes (SQLite BLOB, Postgres BYTEA) or
 *  pgvector's text form `[0.1,0.2,…]`. Bytes need `dims` to tell int8 from
 *  float32; without it they are read as float32. */
export function decodeVector(raw: SqlValue, dims?: number): number[] {
  if (raw instanceof Uint8Array) {
    const d = dims ?? raw.byteLength / 4;
    const view = blobView(raw, d);
    if (view instanceof Float32Array) return Array.from(view);
    const scale = new DataView(raw.buffer, raw.byteOffset, raw.byteLength).getFloat32(d, true);
    return Array.from(view, (x) => x * scale);
  }
  if (typeof raw === 'string') return raw.replace(/^\[|\]$/g, '').split(',').map(Number);
  throw new Error(`agent-graph: cannot decode a stored vector of type ${typeof raw}`);
}

/** The parameter to bind for a vector on this store. */
function vectorParam(ctx: HandleContext, v: number[]): SqlParam {
  return ctx.store.driver.features.vector ? `[${v.join(',')}]` : encodeInt8(v);
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / Math.sqrt(na * nb);
}

/** Cosine of a query (with its norm) against a stored view — the hot loop. */
function cosineView(q: Float32Array, qNorm: number, v: Int8Array | Float32Array): number {
  let dot = 0, nv = 0;
  for (let i = 0; i < q.length; i++) {
    const x = v[i];
    dot += q[i] * x;
    nv += x * x;
  }
  if (qNorm === 0 || nv === 0) return 0;
  return dot / (qNorm * Math.sqrt(nv));
}

/** The best `k` (score desc, then id asc — deterministic), kept as they come. */
class TopK {
  readonly items: Array<{ id: string; score: number }> = [];
  constructor(private readonly k: number) {}
  private static before(a: { id: string; score: number }, b: { id: string; score: number }): boolean {
    return a.score > b.score || (a.score === b.score && a.id < b.id);
  }
  offer(id: string, score: number): void {
    const items = this.items;
    const c = { id, score };
    if (items.length >= this.k && !TopK.before(c, items[items.length - 1])) return;
    let lo = 0, hi = items.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (TopK.before(items[mid], c)) lo = mid + 1;
      else hi = mid;
    }
    items.splice(lo, 0, c);
    if (items.length > this.k) items.pop();
  }
}

// ---------------------------------------------------------------------------
// Writing vectors

function checkVectors(cfg: EmbeddingConfig, texts: string[], vectors: unknown): number[][] {
  if (!Array.isArray(vectors) || vectors.length !== texts.length) {
    throw new Error(`embed() returned ${Array.isArray(vectors) ? vectors.length : typeof vectors} vectors for ${texts.length} texts`);
  }
  for (const v of vectors) {
    if (!Array.isArray(v) || v.length !== cfg.dims || !v.every((x) => typeof x === 'number' && Number.isFinite(x))) {
      throw new Error(`embed() returned a vector that is not ${cfg.dims} finite numbers`);
    }
  }
  return vectors as number[][];
}

/**
 * Embed the given nodes (their given versions) under the configured rule, in
 * batches. A node the rule declines has its vector removed, so the vector
 * table never says more than the rule allows. Throws on a failed model call
 * after committing the batches before it — the caller decides whether that
 * is fatal (`reembed`: yes; `put`: no, log it).
 */
export async function embedNodes(ctx: HandleContext, nodes: NodeRecord[]): Promise<ReembedResult> {
  const cfg = ctx.embedding;
  if (!cfg) throw new Error('agent-graph: no embedding configured on this graph (open it with { embedding })');
  const out: ReembedResult = { embedded: 0, cleared: 0, skipped: 0 };
  const selected: Array<{ node: NodeRecord; text: string }> = [];
  for (const node of nodes) {
    const text = embeddingText(cfg, node);
    if (text === null) {
      const { changes } = await ctx.store.driver.run(SQL.deleteVector, [node.id]);
      if (changes > 0) out.cleared++;
      else out.skipped++;
    } else {
      selected.push({ node, text });
    }
  }
  for (let i = 0; i < selected.length; i += BATCH) {
    const batch = selected.slice(i, i + BATCH);
    const vectors = checkVectors(cfg, batch.map((b) => b.text), await cfg.embed(batch.map((b) => b.text)));
    await ctx.store.transaction(async () => {
      for (let j = 0; j < batch.length; j++) {
        const { node } = batch[j];
        await ctx.store.driver.run(SQL.upsertVector, [node.id, node.version, cfg.dims, vectorParam(ctx, vectors[j])]);
      }
    });
    out.embedded += batch.length;
  }
  return out;
}

/** `Graph.reembed`: the latest version of every node the options select. */
export async function reembed(ctx: HandleContext, opts: ReembedOptions = {}): Promise<ReembedResult> {
  const where: string[] = [];
  const params: SqlParam[] = [];
  if (opts.kinds && opts.kinds.length) {
    where.push(`kind IN (${opts.kinds.map(() => '?').join(', ')})`);
    params.push(...opts.kinds);
  }
  if (opts.since !== undefined) {
    where.push('recorded_at >= ?');
    params.push(opts.since);
  }
  const rows = await ctx.store.driver.all(`SELECT * FROM nodes${where.length ? ' WHERE ' + where.join(' AND ') : ''} ORDER BY id`, params);
  return embedNodes(ctx, rows.map((r) => rowToNode(r as SqlRow)));
}

/** Vector rows `compactVectors` reads per page — bounds what one page holds. */
const COMPACT_PAGE = 500;

/**
 * `Graph.compactVectors`: the stored vectors brought in line with the current
 * rule, with no model call. A node the rule now excludes loses its vector; a
 * float32 BLOB (written before int8) is rewritten as int8 from its own values
 * — same version, same dims, so nothing it ranks changes beyond the measured
 * int8 rounding. Paged by id, one transaction per page.
 */
export async function compactVectors(ctx: HandleContext): Promise<CompactVectorsResult> {
  const cfg = ctx.embedding;
  if (!cfg) throw new Error('agent-graph: no embedding configured on this graph (open it with { embedding })');
  const driver = ctx.store.driver;
  const out: CompactVectorsResult = { kept: 0, converted: 0, cleared: 0 };
  let after = '';
  for (;;) {
    const rows = await driver.all(
      `SELECT n.*, v.dims AS v_dims, v.embedding AS v_embedding FROM node_vectors v JOIN nodes n ON n.id = v.id WHERE v.id > ? ORDER BY v.id LIMIT ?`,
      [after, COMPACT_PAGE],
    );
    if (rows.length === 0) break;
    after = rows[rows.length - 1].id as string;
    await ctx.store.transaction(async () => {
      for (const r of rows) {
        const node = rowToNode(r as SqlRow);
        if (embeddingText(cfg, node) === null) {
          await driver.run(SQL.deleteVector, [node.id]);
          out.cleared++;
          continue;
        }
        const raw = r.v_embedding;
        const dims = num(r.v_dims);
        if (!driver.features.vector && raw instanceof Uint8Array && raw.byteLength === dims * 4) {
          await driver.run('UPDATE node_vectors SET embedding = ? WHERE id = ?', [encodeInt8(decodeVector(raw, dims)), node.id]);
          out.converted++;
        } else {
          out.kept++;
        }
      }
    });
    if (rows.length < COMPACT_PAGE) break;
  }
  // A vector whose node row is gone cannot be judged by the rule; there is none
  // in a graph this engine wrote (nodes are never deleted), so it is not looked for.
  return out;
}

// ---------------------------------------------------------------------------
// Searching

/**
 * Nearest node ids to `text`, most similar first, among nodes of `kinds` (or
 * any) that exist at `asOf`. The query text goes through the same `embed`
 * as the nodes did — a different model would make the distances meaningless.
 */
export async function semanticLocate(
  ctx: HandleContext,
  text: string,
  opts: { kinds?: string[]; asOf?: number; limit: number },
): Promise<NodeId[]> {
  const cfg = ctx.embedding;
  if (!cfg) throw new Error('agent-graph: `locate`/`semantic` given but no embedding (or locator) is configured on this graph');
  const [query] = checkVectors(cfg, [text], await cfg.embed([text]));
  const kindFilter = opts.kinds && opts.kinds.length ? ` AND n.kind IN (${opts.kinds.map(() => '?').join(', ')})` : '';
  const kindParams: SqlParam[] = opts.kinds && opts.kinds.length ? [...opts.kinds] : [];
  // Over-fetch: some candidates may not exist at `asOf` and fall out below.
  const fetch = opts.asOf === undefined ? opts.limit : opts.limit * 4;

  let ranked: NodeId[];
  if (ctx.store.driver.features.vector) {
    const rows = await ctx.store.driver.all(
      `SELECT v.id FROM node_vectors v JOIN nodes n ON n.id = v.id WHERE v.dims = ?${kindFilter} ORDER BY v.embedding <=> ?::vector LIMIT ?`,
      [cfg.dims, ...kindParams, `[${query.join(',')}]`, fetch],
    );
    ranked = rows.map((r) => r.id as string);
  } else {
    const q = Float32Array.from(query);
    let qNorm = 0;
    for (let i = 0; i < q.length; i++) qNorm += q[i] * q[i];
    qNorm = Math.sqrt(qNorm);
    const top = new TopK(fetch);
    const visit = (r: SqlRow): void => {
      const raw = r.embedding;
      if (!(raw instanceof Uint8Array)) throw new Error(`agent-graph: a stored vector of type ${typeof raw} on a BLOB store`);
      top.offer(r.id as string, cosineView(q, qNorm, blobView(raw, cfg.dims)));
    };
    const sql = `SELECT v.id, v.embedding FROM node_vectors v JOIN nodes n ON n.id = v.id WHERE v.dims = ?${kindFilter}`;
    const params = [cfg.dims, ...kindParams];
    const driver = ctx.store.driver;
    if (driver.each) await driver.each(sql, params, visit);
    else for (const r of await driver.all(sql, params)) visit(r);
    ranked = top.items.map((r) => r.id);
  }
  if (opts.asOf === undefined) return ranked.slice(0, opts.limit);
  const out: NodeId[] = [];
  for (const id of ranked) {
    if (await ctx.store.nodeAt(id, opts.asOf)) out.push(id);
    if (out.length >= opts.limit) break;
  }
  return out;
}

/** Vector rows of a graph, for tests and tooling: id, version, dims. */
export async function vectorIndex(ctx: HandleContext): Promise<Array<{ id: NodeId; version: number; dims: number }>> {
  const rows = await ctx.store.driver.all(`SELECT id, version, dims FROM node_vectors ORDER BY id`);
  return rows.map((r) => ({ id: r.id as string, version: num(r.version), dims: num(r.dims) }));
}

// ---------------------------------------------------------------------------
// The one built-in provider

export interface OpenAiCompatibleOptions {
  /** Server root, e.g. `https://api.openai.com`, `http://localhost:11434`
   *  (Ollama), `http://vllm:8000`. `/v1/embeddings` is appended; a base that
   *  already ends in `/v1` is accepted. */
  baseUrl: string;
  apiKey?: string;
  model: string;
  dims: number;
  /** Test seam. Default: global fetch. */
  fetch?: typeof fetch;
}

/**
 * `embed()` against any server speaking the OpenAI embeddings shape —
 * OpenAI, Ollama, vLLM, LM Studio, text-embeddings-inference. No SDK: one
 * POST, one JSON body. Returns the vectors in input order (the API returns
 * them indexed) and refuses a response whose dimension is not `dims`, so a
 * misconfigured model cannot quietly fill the table with incomparable rows.
 */
export function openAiCompatibleEmbedder(opts: OpenAiCompatibleOptions): Pick<EmbeddingConfig, 'embed' | 'dims'> {
  const root = opts.baseUrl.replace(/\/+$/, '');
  const url = `${root}${root.endsWith('/v1') ? '' : '/v1'}/embeddings`;
  const doFetch = opts.fetch ?? fetch;
  return {
    dims: opts.dims,
    async embed(texts) {
      if (texts.length === 0) return [];
      const headers: Record<string, string> = { 'content-type': 'application/json' };
      if (opts.apiKey) headers.authorization = `Bearer ${opts.apiKey}`;
      const res = await doFetch(url, { method: 'POST', headers, body: JSON.stringify({ model: opts.model, input: texts }) });
      if (!res.ok) {
        // The body may echo our input; the status and model are enough to act on.
        throw new Error(`embeddings request failed: HTTP ${res.status} from ${url} (model ${opts.model})`);
      }
      const body = (await res.json()) as { data?: Array<{ index?: number; embedding?: unknown }> };
      if (!Array.isArray(body.data)) throw new Error(`embeddings response has no data array (${url})`);
      const out: number[][] = new Array(texts.length);
      body.data.forEach((d, i) => {
        const at = typeof d.index === 'number' ? d.index : i;
        out[at] = d.embedding as number[];
      });
      for (let i = 0; i < out.length; i++) {
        if (!Array.isArray(out[i])) throw new Error(`embeddings response is missing vector #${i}`);
        if (out[i].length !== opts.dims) throw new Error(`model ${opts.model} returned ${out[i].length} dimensions, configured dims is ${opts.dims}`);
      }
      return out;
    },
  };
}
