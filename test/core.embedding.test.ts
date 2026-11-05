/** The embedding suite over SQLite (int8 BLOBs, cosine in JS), plus the
 *  pieces that need no database: the text rules, the BLOB encodings and the
 *  HTTP provider. */
import assert from 'node:assert/strict';
import { after, describe, test } from 'node:test';
import { openAiCompatibleEmbedder, textRuleFrom } from '../src/index.js';
import { cosine, decodeVector, encodeFloat32, encodeInt8 } from '../src/embedding.js';
import { DIMS, embeddingSuite, fakeEmbedder, fakeVector } from './suites/embedding.suite.js';
import { sqliteHarness } from './suites/harness.js';

const h = sqliteHarness();
embeddingSuite(h);
after(() => h.cleanup());

const node = { id: 'n', kind: 'note', label: 'Rounding', attrs: { text: 'half-up', tags: ['a'], skip: null }, origin: 'o', provenance: 'claimed' as const, version: 1, recordedAt: 0, createdBy: 'o', createdAt: 0, flags: [] };

describe('text rules', () => {
  test("'label' | 'label+attrs' | { attrs }", () => {
    assert.equal(textRuleFrom(undefined)(node), 'Rounding');
    assert.equal(textRuleFrom('label')(node), 'Rounding');
    assert.equal(textRuleFrom('label+attrs')(node), 'Rounding\ntext: half-up\ntags: ["a"]', 'strings verbatim, other values as JSON, nulls dropped');
    assert.equal(textRuleFrom({ attrs: ['text', 'missing'] })(node), 'Rounding\ntext: half-up');
  });

  test('float32 round trip and cosine', () => {
    const v = [0.5, -1, 2, 0.25];
    assert.deepEqual(decodeVector(encodeFloat32(v)), v);
    assert.deepEqual(decodeVector('[0.5,-1,2,0.25]'), v, "pgvector's text form decodes too");
    assert.equal(cosine([1, 0], [1, 0]), 1);
    assert.equal(cosine([1, 0], [0, 1]), 0);
    assert.equal(cosine([0, 0], [1, 1]), 0, 'a zero vector is nowhere');
  });
});

describe('BLOB encoding: int8, older float32 still read', () => {
  test('int8 is dims + 4 bytes and decodes to within one step of the scale', () => {
    const v = [0.5, -1, 2, 0.25, 0];
    const b = encodeInt8(v);
    assert.equal(b.byteLength, v.length + 4, 'a quarter of float32, plus the scale');
    const back = decodeVector(b, v.length);
    const step = 2 / 127;
    back.forEach((x, i) => assert.ok(Math.abs(x - v[i]) <= step / 2 + 1e-6, `dim ${i}: ${x} vs ${v[i]}`));
    assert.deepEqual(decodeVector(encodeInt8([0, 0, 0]), 3), [0, 0, 0], 'a zero vector stays zero (scale 0)');
  });

  test('length tells the two apart, so a float32 blob of the same dims still decodes', () => {
    const v = [0.5, -1, 2, 0.25];
    assert.deepEqual(decodeVector(encodeFloat32(v), 4), v);
    assert.throws(() => decodeVector(new Uint8Array(7), 4), /neither int8 nor float32 of 4 dims/);
  });

  test('a graph writes int8 blobs and still ranks a float32 blob written before', async () => {
    const g = await h.open({ origin: 'rt', trusted: true, embedding: fakeEmbedder() });
    await g.put({ id: 'n:pricing', kind: 'note', label: 'rounding rule in the pricing service is half-up' });
    await g.put({ id: 'n:checkout', kind: 'note', label: 'coupon applied at checkout before tax' });
    const ctx = (g as unknown as { ctx(): { store: { driver: { all(s: string, p?: unknown[]): Promise<Array<Record<string, unknown>>>; run(s: string, p?: unknown[]): Promise<unknown> } } } }).ctx();
    const rows = await ctx.store.driver.all('SELECT id, embedding FROM node_vectors ORDER BY id');
    assert.deepEqual(rows.map((r) => (r.embedding as Uint8Array).byteLength), [DIMS + 4, DIMS + 4]);
    // An older graph: this node's vector was written as float32.
    await g.put({ id: 'n:old', kind: 'note', label: 'pricing rounding half-up, from before' });
    await ctx.store.driver.run('UPDATE node_vectors SET embedding = ? WHERE id = ?', [encodeFloat32(fakeVector('pricing rounding half-up, from before')), 'n:old']);
    const m = await g.match({ semantic: 'pricing rounding half-up from before', limit: 2 });
    assert.deepEqual(m.map((n) => n.id), ['n:old', 'n:pricing'], 'the float32 row ranks alongside the int8 ones');
  });

  test('compactVectors rewrites a float32 blob as int8 from its own values', async () => {
    const emb = fakeEmbedder();
    const g = await h.open({ origin: 'rt', trusted: true, embedding: emb });
    await g.put({ id: 'n:old', kind: 'note', label: 'pricing rounding half-up' });
    const ctx = (g as unknown as { ctx(): { store: { driver: { all(s: string, p?: unknown[]): Promise<Array<Record<string, unknown>>>; run(s: string, p?: unknown[]): Promise<unknown> } } } }).ctx();
    const v = fakeVector('pricing rounding half-up');
    await ctx.store.driver.run('UPDATE node_vectors SET embedding = ? WHERE id = ?', [encodeFloat32(v), 'n:old']);
    const calls = emb.calls.length;
    assert.deepEqual(await g.compactVectors(), { kept: 0, converted: 1, cleared: 0 });
    assert.equal(emb.calls.length, calls, 'no model call');
    const [row] = await ctx.store.driver.all('SELECT version, dims, embedding FROM node_vectors');
    assert.equal((row.embedding as Uint8Array).byteLength, DIMS + 4);
    assert.equal(row.version, 1, 'same version');
    assert.ok(cosine(decodeVector(row.embedding as Uint8Array, DIMS), v) > 0.999, 'the same direction');
  });

  test('int8 ranks like float32 and ties break by id', async () => {
    const g = await h.open({ origin: 'rt', trusted: true, embedding: fakeEmbedder() });
    // Two nodes with the same label have the same vector: equal scores.
    await g.put({ id: 'n:b', kind: 'note', label: 'pricing rounding' });
    await g.put({ id: 'n:a', kind: 'note', label: 'pricing rounding' });
    await g.put({ id: 'n:c', kind: 'note', label: 'checkout coupon tax' });
    const m = await g.match({ semantic: 'pricing rounding', limit: 3 });
    assert.deepEqual(m.map((n) => n.id), ['n:a', 'n:b', 'n:c']);
    const one = await g.match({ semantic: 'pricing rounding', limit: 1 });
    assert.deepEqual(one.map((n) => n.id), ['n:a'], 'a bounded top-k keeps the same order as a full sort');
  });
});

describe('openAiCompatibleEmbedder', () => {
  test('posts to /v1/embeddings, sends the key, returns vectors in input order, refuses wrong dims', async () => {
    const seen: Array<{ url: string; init: RequestInit }> = [];
    const fetchStub = (async (url: string | URL | Request, init?: RequestInit) => {
      seen.push({ url: String(url), init: init! });
      const body = JSON.parse(String(init!.body)) as { model: string; input: string[] };
      // Out of order on purpose: the client must use `index`.
      const data = body.input.map((t, i) => ({ index: i, embedding: [t.length, 1] })).reverse();
      return new Response(JSON.stringify({ data, model: body.model }), { status: 200 });
    }) as typeof fetch;
    const e = openAiCompatibleEmbedder({ baseUrl: 'http://model:11434/', apiKey: 'k', model: 'm', dims: 2, fetch: fetchStub });
    assert.deepEqual(await e.embed(['ab', 'abcd']), [[2, 1], [4, 1]]);
    assert.equal(seen[0].url, 'http://model:11434/v1/embeddings');
    assert.equal((seen[0].init.headers as Record<string, string>).authorization, 'Bearer k');
    assert.deepEqual(await e.embed([]), [], 'no request for no texts');
    assert.equal(seen.length, 1);
    const v1 = openAiCompatibleEmbedder({ baseUrl: 'https://api.openai.com/v1', model: 'm', dims: 2, fetch: fetchStub });
    await v1.embed(['x']);
    assert.equal(seen[1].url, 'https://api.openai.com/v1/embeddings', 'a base that already ends in /v1 is not doubled');
    assert.equal((seen[1].init.headers as Record<string, string>).authorization, undefined, 'no key, no header');

    const wrong = openAiCompatibleEmbedder({ baseUrl: 'http://m', model: 'm', dims: 3, fetch: fetchStub });
    await assert.rejects(wrong.embed(['x']), /returned 2 dimensions, configured dims is 3/);
    const down = openAiCompatibleEmbedder({ baseUrl: 'http://m', model: 'm', dims: 2, fetch: (async () => new Response('nope', { status: 503 })) as typeof fetch });
    await assert.rejects(down.embed(['x']), /HTTP 503/);
  });
});
