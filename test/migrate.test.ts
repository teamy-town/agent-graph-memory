/**
 * Moving graphs from Postgres to SQLite (migrate.ts + the server's
 * --migrate-from-pg). Needs a Postgres server; skipped — not failed — without
 * TEST_PG_URL (see test/postgres.test.ts for how to start one).
 */
import assert from 'node:assert/strict';
import { spawn, type ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { existsSync, mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { after, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { PostgresDriver } from '../src/drivers/postgres.js';
import { dropGraph, openGraph } from '../src/index.js';
import { MIGRATED_FROM, migrateSchemaToSqlite, unmigratedSchemas } from '../src/migrate.js';
import { schemaFor } from '../src/server.js';
import { DIMS, fakeVector } from './suites/embedding.suite.js';
import type { Graph } from '../src/types.js';

const url = process.env.TEST_PG_URL;
const BIN = fileURLToPath(new URL('../src/server.js', import.meta.url));
const embedding = { embed: async (texts: string[]) => texts.map((t) => fakeVector(t)), dims: DIMS, text: (n: { label: string }) => n.label };

/** A Postgres graph with every table populated, vectors included. */
async function seed(schema: string): Promise<void> {
  const g = await openGraph({ driver: 'postgres', connectionString: url!, schema }, { origin: 'rt', trusted: true, privileged: true, embedding });
  await g.put({ id: 'ticket:1', kind: 'ticket', label: 'Release failed on arm64', provenance: 'observed' });
  await g.put({ id: 'ticket:1', kind: 'ticket', label: 'Release failed on arm64 builds', provenance: 'observed' }); // a second version
  await g.put({ id: 'file:a.js', kind: 'file', label: 'a.js', provenance: 'observed' });
  await g.put({ id: 'note:1', kind: 'note', label: 'The arm64 job ran out of memory', attrs: { text: 'x' } });
  const e = await g.link({ src: 'ticket:1', dst: 'file:a.js', rel: 'touched', provenance: 'observed' });
  await g.link({ src: 'note:1', dst: 'ticket:1', rel: 'about' });
  await g.supersede(e.id, { src: 'ticket:1', dst: 'file:a.js', rel: 'touched', cost: 2, provenance: 'observed' });
  await g.recall({ seeds: ['ticket:1'], maxCost: 2 }); // writes `access` rows
  await g.close();
}

async function rows(schema: string): Promise<Record<string, number>> {
  const d = await PostgresDriver.open({ driver: 'postgres', connectionString: url! });
  const out: Record<string, number> = {};
  for (const t of ['nodes', 'node_versions', 'edges', 'access', 'node_vectors']) {
    out[t] = Number((await d.all(`SELECT COUNT(*) AS n FROM "${schema}".${t}`))[0].n);
  }
  await d.close();
  return out;
}

async function schemaExists(schema: string): Promise<boolean> {
  const d = await PostgresDriver.open({ driver: 'postgres', connectionString: url! });
  const r = await d.all(`SELECT 1 AS x FROM pg_namespace WHERE nspname = ?`, [schema]);
  await d.close();
  return r.length > 0;
}

/** What a reader sees: every node, every edge, a recall, a semantic search. */
async function view(g: Graph) {
  const dump = await g.dump();
  const recall = await g.recall({ seeds: ['ticket:1'], maxCost: 3 });
  const near = await g.match({ semantic: 'arm64 out of memory', limit: 2 });
  return {
    nodes: dump.nodes, versions: dump.nodeVersions, edges: dump.edges, access: dump.access,
    recall: recall.hits.map((h) => [h.node.id, h.cost]),
    near: near.map((n) => n.id),
  };
}

describe('move a graph from Postgres to SQLite', { skip: url ? false : 'TEST_PG_URL not set' }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-graph-migrate-'));
  const schemas: string[] = [];
  const fresh = () => { const s = `mig_${process.pid}_${randomBytes(3).toString('hex')}`; schemas.push(s); return s; };
  after(async () => {
    for (const s of schemas) await dropGraph({ driver: 'postgres', connectionString: url!, schema: s }).catch(() => {});
    rmSync(dir, { recursive: true, force: true });
  });

  test('copies every row, vectors included, and the copy reads the same', async () => {
    const schema = fresh();
    await seed(schema);
    const before = await rows(schema);
    assert.ok(before.access > 0 && before.node_vectors > 0 && before.node_versions > before.nodes, JSON.stringify(before));
    const file = join(dir, 'g1.sqlite');
    const r = await migrateSchemaToSqlite(url!, schema, file);
    assert.equal(r.status, 'migrated');
    assert.deepEqual(r.status === 'migrated' && r.counts, { ...before });
    assert.deepEqual(readdirSync(dir).filter((f) => f.startsWith('g1')), ['g1.sqlite'], 'no temp or WAL file left beside it');

    const pg = await openGraph({ driver: 'postgres', connectionString: url!, schema }, { embedding });
    const lite = await openGraph(file, { embedding });
    assert.deepEqual(await view(lite), await view(pg));
    await pg.close();
    await lite.close();

    const db = new DatabaseSync(file, { readOnly: true });
    const v = db.prepare('SELECT dims, length(embedding) AS bytes FROM node_vectors LIMIT 1').get() as { dims: number; bytes: number };
    assert.equal(v.bytes, v.dims + 4, 'vectors are stored int8 (dims + scale), from the row\'s own dims');
    assert.equal((db.prepare('SELECT value FROM meta WHERE key = ?').get(MIGRATED_FROM) as { value: string }).value, schema);
    db.close();
    assert.deepEqual(await rows(schema), before, 'the source schema is unchanged');
  });

  test('a schema that does not exist is not created, and no file is written', async () => {
    const schema = `mig_absent_${randomBytes(3).toString('hex')}`;
    const file = join(dir, 'absent.sqlite');
    assert.equal((await migrateSchemaToSqlite(url!, schema, file)).status, 'absent');
    assert.equal(existsSync(file), false);
    assert.equal(await schemaExists(schema), false);
  });

  test('an existing file is never overwritten', async () => {
    const schema = fresh();
    await seed(schema);
    const file = join(dir, 'taken.sqlite');
    (await openGraph(file)).close();
    await assert.rejects(migrateSchemaToSqlite(url!, schema, file), /already exists/);
  });

  test('a failed copy leaves no file behind and the source untouched', async () => {
    const schema = fresh();
    await seed(schema);
    // A vector whose bytes disagree with its dims cannot be re-encoded.
    const d = await PostgresDriver.open({ driver: 'postgres', connectionString: url! });
    await d.run(`UPDATE "${schema}".node_vectors SET dims = dims + 1 WHERE id = 'note:1'`);
    await d.close();
    const before = await rows(schema);
    const file = join(dir, 'broken.sqlite');
    await assert.rejects(migrateSchemaToSqlite(url!, schema, file), /note:1/);
    assert.deepEqual(readdirSync(dir).filter((f) => f.startsWith('broken')), []);
    assert.deepEqual(await rows(schema), before);
  });
});

interface Spawned { child: ChildProcess; port: number; stderr: string[] }
function start(args: string[]): Promise<Spawned> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [BIN, ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
    const stderr: string[] = [];
    let out = '';
    child.stderr!.setEncoding('utf8').on('data', (d: string) => stderr.push(d));
    child.stdout!.setEncoding('utf8').on('data', (d: string) => {
      out += d;
      const nl = out.indexOf('\n');
      if (nl >= 0) resolve({ child, port: (JSON.parse(out.slice(0, nl)) as { listening: { port: number } }).listening.port, stderr });
    });
    child.on('exit', (code) => reject(new Error(`exited ${code}: ${stderr.join('')}`)));
  });
}
function listUnmigrated(dir: string): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [BIN, '--driver', 'sqlite', '--sqlite-dir', dir, '--migrate-from-pg', url!, '--list-unmigrated'], { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    child.stdout!.setEncoding('utf8').on('data', (d: string) => (out += d));
    child.on('exit', (code) => (code === 0 ? resolve(JSON.parse(out) as string[]) : reject(new Error(`exit ${code}`))));
  });
}

describe('agent-graph-server --migrate-from-pg', { skip: url ? false : 'TEST_PG_URL not set' }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-graph-migrate-srv-'));
  // A long id: its schema name is shortened with a hash, so only the server,
  // which knows the id, can find it.
  const LONG = `acct-${randomBytes(20).toString('hex')}:project-x:store-${randomBytes(8).toString('hex')}`;
  const BROKEN = `broken_${process.pid}_${randomBytes(3).toString('hex')}`;
  const created = [schemaFor(LONG), BROKEN];
  let s: Spawned;
  const post = async (op: string, body: Record<string, unknown>) => {
    const r = await fetch(`http://127.0.0.1:${s.port}/graph/${op}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    return { status: r.status, json: (await r.json()) as Record<string, unknown> };
  };
  after(async () => {
    if (s) s.child.kill('SIGTERM');
    for (const schema of created) await dropGraph({ driver: 'postgres', connectionString: url!, schema }).catch(() => {});
    rmSync(dir, { recursive: true, force: true });
  });

  test('first open moves the graph; a failed move is served from Postgres and said loudly; drop does not resurrect', async () => {
    assert.ok(LONG.length > 63 && schemaFor(LONG) !== LONG);
    await seed(schemaFor(LONG));
    await seed(BROKEN);
    const d = await PostgresDriver.open({ driver: 'postgres', connectionString: url! });
    await d.run(`UPDATE "${BROKEN}".node_vectors SET dims = dims + 1 WHERE id = 'note:1'`);
    await d.close();
    const pending = await listUnmigrated(dir);
    assert.ok(pending.includes(schemaFor(LONG)) && pending.includes(BROKEN));

    s = await start(['--driver', 'sqlite', '--sqlite-dir', dir, '--migrate-from-pg', url!, '--port', '0']);
    const st = await post('stats', { graphId: LONG });
    assert.equal(st.status, 200, JSON.stringify(st.json));
    assert.equal(st.json.nodes, 3);
    assert.ok(existsSync(join(dir, `${LONG}.sqlite`)), 'the file is named by the graph id, not the shortened schema');
    assert.match(s.stderr.join(''), /moved from Postgres schema .* row counts verified/);

    const b = await post('stats', { graphId: BROKEN });
    assert.equal(b.status, 200);
    assert.equal(b.json.nodes, 3, 'never served empty');
    assert.equal(existsSync(join(dir, `${BROKEN}.sqlite`)), false);
    assert.match(s.stderr.join(''), /MIGRATION FAILED/);

    const left = await listUnmigrated(dir);
    assert.ok(!left.includes(schemaFor(LONG)) && left.includes(BROKEN));

    // A brand-new graph is a fresh file; nothing is created on Postgres.
    const NEW = `new_${randomBytes(4).toString('hex')}`;
    assert.equal((await post('put', { graphId: NEW, origin: 'rt', id: 'x:1', kind: 'x', label: 'X' })).status, 200);
    assert.equal(await schemaExists(NEW), false);

    // Dropping a moved graph leaves an empty file that keeps it from coming back.
    assert.equal((await post('drop', { graphId: LONG, confirm: true })).status, 200);
    s.child.kill('SIGTERM');
    s = await start(['--driver', 'sqlite', '--sqlite-dir', dir, '--migrate-from-pg', url!, '--port', '0']);
    assert.equal((await post('stats', { graphId: LONG })).json.nodes, 0);
    assert.ok(!(await listUnmigrated(dir)).includes(schemaFor(LONG)));
  });
});
