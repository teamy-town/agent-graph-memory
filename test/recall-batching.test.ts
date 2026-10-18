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
