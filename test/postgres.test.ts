/**
 * The same suites over Postgres. Skipped — not failed — without TEST_PG_URL,
 * because a unit test run must not depend on a database server. To run them:
 *
 *   docker run --rm -d --name ag-pg-test -e POSTGRES_PASSWORD=t -p 15432:5432 pgvector/pgvector:pg16
 *   TEST_PG_URL=postgres://postgres:t@localhost:15432/postgres npm test
 *   docker rm -f ag-pg-test
 *
 * Each graph gets its own schema (dropped afterwards), so the suites may run
 * in parallel processes against one server.
 */
import assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { after, describe, test } from 'node:test';
import { renderSql } from '../src/index.js';
import { PostgresDriver } from '../src/drivers/postgres.js';
import { initSchema } from '../src/schema.js';
import { embeddingSuite } from './suites/embedding.suite.js';
import { graphSuite } from './suites/graph.suite.js';
import { postgresHarness } from './suites/harness.js';
import { journalSuite } from './suites/journal.suite.js';
import { recallSuite } from './suites/recall.suite.js';

const url = process.env.TEST_PG_URL;

describe('renderSql (no server needed)', () => {
  test('placeholders, OR IGNORE, pseudo-types, instr, literals untouched', () => {
    assert.equal(renderSql('SELECT ? FROM t WHERE a = ? AND b = ?', 'postgres'), 'SELECT $1 FROM t WHERE a = $2 AND b = $3');
    assert.equal(renderSql("SELECT '?' FROM t WHERE a = ?", 'postgres'), "SELECT '?' FROM t WHERE a = $1", 'a ? inside a literal is not a placeholder');
    assert.equal(renderSql('INSERT OR IGNORE INTO t (a) VALUES (?)', 'postgres'), 'INSERT INTO t (a) VALUES ($1) ON CONFLICT DO NOTHING');
    assert.equal(renderSql('x EPOCH, y VECTOR_TYPE, z REAL, w TEXT NOT NULL', 'postgres'), 'x BIGINT, y BYTEA, z DOUBLE PRECISION, w TEXT COLLATE "C" NOT NULL');
    assert.equal(renderSql('y VECTOR_TYPE', 'postgres', { vector: true }), 'y vector');
    assert.equal(renderSql('x EPOCH, y VECTOR_TYPE, z REAL', 'sqlite'), 'x INTEGER, y BLOB, z REAL');
    assert.equal(renderSql('instr(lower(a), lower(?)) > 0', 'postgres'), 'strpos(lower(a), lower($1)) > 0');
    assert.equal(renderSql('CREATE INDEX IF NOT EXISTS nodes_label ON nodes LABEL_INDEX_METHOD (label)', 'postgres'),
      'CREATE INDEX IF NOT EXISTS nodes_label ON nodes USING hash (label)');
    assert.equal(renderSql('CREATE INDEX IF NOT EXISTS nodes_label ON nodes LABEL_INDEX_METHOD (label)', 'sqlite'),
      'CREATE INDEX IF NOT EXISTS nodes_label ON nodes  (label)');
    const once = renderSql('INSERT OR IGNORE INTO t (a) VALUES (?)', 'postgres');
    assert.equal(renderSql(once, 'postgres'), once, 'idempotent');
  });
});

if (!url) {
  test('postgres suites', { skip: 'TEST_PG_URL is not set' }, () => {});
} else {
  const h = postgresHarness(url);
  describe('postgres', () => {
    test('driver: schema per graph, pgvector detected, BIGINT epochs come back as numbers', async () => {
      const d = await PostgresDriver.open({ driver: 'postgres', connectionString: url, schema: `agt_probe_${process.pid}` });
      try {
        assert.equal(d.dialect, 'postgres');
        assert.equal(d.features.vector, true, 'the test image ships pgvector');
        await d.exec('CREATE TABLE IF NOT EXISTS probe (t EPOCH, n INTEGER)');
        await d.run('INSERT INTO probe (t, n) VALUES (?, ?)', [Date.UTC(2026, 8, 1), 7]);
        const [row] = await d.all('SELECT t, n, COUNT(*) OVER () AS c FROM probe');
        assert.equal(row.t, Date.UTC(2026, 8, 1));
        assert.equal(typeof row.t, 'number');
        assert.equal(row.c, 1);
        assert.deepEqual(await d.columns('probe'), ['t', 'n']);
        assert.ok((await d.storageBytes()) > 0);
        assert.deepEqual(await d.all(`SELECT current_schema() AS s`), [{ s: `agt_probe_${process.pid}` }]);
      } finally {
        await d.destroy();
      }
    });
    test('existing btree label indexes migrate without losing long labels or exact lookup', async () => {
      const d = await PostgresDriver.open({ driver: 'postgres', connectionString: url,
        schema: `agt_long_${process.pid}_${randomBytes(3).toString('hex')}` });
      try {
        await initSchema(d);
        await d.exec('DROP INDEX nodes_label; DROP INDEX node_versions_label; '
          + 'CREATE INDEX nodes_label ON nodes (label); CREATE INDEX node_versions_label ON node_versions (label)');
        const label = randomBytes(3000).toString('hex');
        const versionSql = 'INSERT INTO node_versions (id, version, kind, label, attrs, origin, provenance, recorded_at, flags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)';
        await assert.rejects(d.run(versionSql, ['note:long', 1, 'note', label, '{}', 'test', 'claimed', 1, '[]']),
          /index row size .* exceeds btree/);
        await initSchema(d);
        const indexes = await d.all("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname IN ('nodes_label', 'node_versions_label') ORDER BY indexname");
        assert.equal(indexes.length, 2);
        for (const row of indexes) assert.match(String(row.indexdef), /USING hash \(label\)/);
        await d.run(versionSql, ['note:long', 1, 'note', label, '{}', 'test', 'claimed', 1, '[]']);
        await d.run('INSERT INTO nodes (id, kind, label, attrs, origin, provenance, version, recorded_at, created_by, created_at, flags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
          ['note:long', 'note', label, '{}', 'test', 'claimed', 1, 1, 'test', 1, '[]']);
        assert.equal((await d.all('SELECT id FROM nodes WHERE label = ?', [label]))[0]?.id, 'note:long');
      } finally {
        await d.destroy();
      }
    });
    graphSuite(h);
    recallSuite(h);
    journalSuite(h);
    embeddingSuite(h);
  });
  after(() => h.cleanup());
}
