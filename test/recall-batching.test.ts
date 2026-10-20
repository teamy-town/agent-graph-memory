import assert from 'node:assert/strict';
import { test } from 'node:test';
import { openGraph } from '../src/index.js';
import { SqliteDriver } from '../src/drivers/sqlite.js';

test('a wide recall batches database reads without changing paths or the returned limit', async () => {
  const g = await openGraph(':memory:', { origin: 'test', trusted: true, readOnly: false });
  try {
    await g.put({ id: 's', kind: 'seed', label: 'seed', provenance: 'observed' });
    for (let i = 0; i < 64; i++) {
      const id = `n:${String(i).padStart(3, '0')}`;
      await g.put({ id, kind: 'leaf', label: id, provenance: 'observed' });
      await g.link({ src: 's', dst: id, rel: 'about', cost: 1, provenance: 'observed' });
    }
    const original = SqliteDriver.prototype.all;
    let calls = 0;
    SqliteDriver.prototype.all = function (...args) { calls++; return original.apply(this, args); };
    let r;
    try {
      r = await g.as('test', { readOnly: true }).recall({ seeds: ['s'], maxCost: 1, direction: 'out', rels: ['about'], limit: 64 });
    } finally { SqliteDriver.prototype.all = original; }
    assert.equal(r.hits.length, 64);
    assert.equal(r.truncated, false);
    for (const hit of r.hits) {
      assert.equal(hit.cost, 1);
      assert.equal(hit.seed, 's');
      assert.equal(hit.path.length, 1);
      assert.equal(hit.path[0].dst, hit.node.id);
    }
    assert.ok(calls <= 12, `wide recall made ${calls} database round trips`);
  } finally { await g.close(); }
});

test('a wide subgraph batches node versions and retains every induced edge', async () => {
  const g = await openGraph(':memory:', { origin: 'test', trusted: true });
  try {
    await g.put({ id: 's', kind: 'seed', label: 'seed', provenance: 'observed' });
    for (let i = 0; i < 64; i++) {
      const id = `leaf:${i}`;
      await g.put({ id, kind: 'leaf', label: id, provenance: 'observed' });
      await g.link({ src: 's', dst: id, rel: 'about', cost: 1, provenance: 'observed' });
    }
    const original = SqliteDriver.prototype.all;
    let calls = 0;
    SqliteDriver.prototype.all = function (...args) { calls++; return original.apply(this, args); };
    let r;
    try { r = await g.subgraph({ seeds: ['s', 'missing'], maxCost: 1, limit: 64, direction: 'out' }); }
    finally { SqliteDriver.prototype.all = original; }
    assert.equal(r.nodes.length, 65, 'the seed is exempt from the result limit');
    assert.equal(r.edges.length, 64);
    assert.deepEqual(r.seeds, ['s']);
    assert.equal(r.truncated, false);
    assert.ok(calls <= 12, `wide subgraph made ${calls} database round trips`);
  } finally { await g.close(); }
});

test('a recall never reads the edges of a node already at its budget (a hub one hop out costs nothing)', async () => {
  const g = await openGraph(':memory:', { origin: 'test', trusted: true, readOnly: false });
  try {
    await g.put({ id: 'lesson', kind: 'lesson', label: 'lesson', provenance: 'observed' });
    await g.put({ id: 'hub', kind: 'member', label: 'hub', provenance: 'observed' });
    await g.link({ src: 'lesson', dst: 'hub', rel: 'about', cost: 1, provenance: 'observed' });
    // The hub has 300 more edges of the same relation: none is within budget.
    for (let i = 0; i < 300; i++) {
      const id = `note:${i}`;
      await g.put({ id, kind: 'note', label: id, provenance: 'observed' });
      await g.link({ src: 'hub', dst: id, rel: 'about', cost: 1, provenance: 'observed' });
    }
    const original = SqliteDriver.prototype.all;
    let edgeRows = 0;
    SqliteDriver.prototype.all = async function (...args) {
      const rows = await original.apply(this, args);
      if (/FROM edges/.test(String(args[0]))) edgeRows += rows.length;
      return rows;
    };
    let r;
    try {
      r = await g.as('test', { readOnly: true }).recall({ seeds: ['lesson'], maxCost: 1, direction: 'both', rels: ['about'], includeSeeds: true, limit: 50 });
    } finally { SqliteDriver.prototype.all = original; }
    // The answer is unchanged: the seed and the hub, the hub one hop away.
    assert.deepEqual(r.hits.map((h) => [h.node.id, h.cost]), [['lesson', 0], ['hub', 1]]);
    assert.equal(r.truncated, false);
    assert.equal(edgeRows, 1, `the walk read ${edgeRows} edge rows; only the seed's one edge can be within budget`);
  } finally { await g.close(); }
});

test('a walk deeper than one hop still reaches everything within its budget', async () => {
  const g = await openGraph(':memory:', { origin: 'test', trusted: true, readOnly: false });
  try {
    for (const id of ['a', 'b', 'c', 'd']) await g.put({ id, kind: 'k', label: id, provenance: 'observed' });
    await g.link({ src: 'a', dst: 'b', rel: 'r', cost: 1, provenance: 'observed' });
    await g.link({ src: 'b', dst: 'c', rel: 'r', cost: 1.5, provenance: 'observed' });
    await g.link({ src: 'c', dst: 'd', rel: 'r', cost: 1, provenance: 'observed' });
    const r = await g.recall({ seeds: ['a'], maxCost: 2.5, direction: 'out' });
    assert.deepEqual(r.hits.map((h) => [h.node.id, h.cost]), [['b', 1], ['c', 2.5]]);
  } finally { await g.close(); }
});
