/**
 * Moving a graph from Postgres (one schema per graph) to SQLite (one file per
 * graph), for a server that switches drivers and keeps its data.
 *
 * Why not `dump()` / `load()`: a dump leaves out `node_vectors` (derived data,
 * but rebuilding it is one paid model call per embedded node, and the process
 * doing the move may hold no model key), and `load()` runs the content guards
 * on every row again, so one old row that a since-tightened guard rejects
 * would keep the whole graph on the old engine. This is a move of the graph's
 * own rows, so it copies them as they are — every table, vectors included,
 * each vector re-encoded in the compact form the SQLite store writes today
 * (`encodeInt8`, from the row's own dims; the same rewrite `compactVectors`
 * makes, without a second pass).
 *
 * Safety, in order:
 *   - the source is only READ: one `REPEATABLE READ READ ONLY` transaction, so
 *     every table is copied from the same instant and nothing can be written;
 *   - the copy goes to `<file>.migrating`, in one SQLite transaction;
 *   - every table's row count in the copy must equal the source's, or the
 *     temp file is deleted and the move fails;
 *   - only then is the temp file renamed onto `<file>` (same directory, so the
 *     rename is atomic). A crash at any point leaves either no file (the next
 *     open migrates again) or the complete one.
 *   - the Postgres schema is never modified or dropped: it stays the backup.
 *
 * The copy records where it came from in the file's `meta` table
 * (`migrated_from` = the schema name), which is how `unmigratedSchemas()`
 * tells which legacy schemas still hold data no file has taken over.
 */

import { existsSync, readdirSync, renameSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { quoteIdent, type SqlParam, type SqlRow } from './driver.js';
import { PostgresDriver } from './drivers/postgres.js';
import { SqliteDriver } from './drivers/sqlite.js';
import { decodeVector, encodeInt8 } from './embedding.js';
import { COLUMNS, PRIMARY_KEY, initSchema, num, type TableName } from './schema.js';

/** The tables a graph's rows live in (meta is copied separately). */
const TABLES = Object.keys(COLUMNS) as TableName[];
/** Rows read from Postgres per page. */
const PAGE = 2000;
/** The `meta` key naming the Postgres schema a file was copied from. */
export const MIGRATED_FROM = 'migrated_from';

export type MigrationResult =
  /** The schema does not exist (or holds no graph tables): nothing to move. */
  | { status: 'absent'; schema: string }
  /** The schema exists but holds no rows: nothing to move. */
  | { status: 'empty'; schema: string }
  | { status: 'migrated'; schema: string; counts: Record<TableName, number>; ms: number };

/** The Postgres driver renders every statement from the SQLite dialect
 *  (`renderSql`), which rewrites a few upper-case words; a schema name holding
 *  one of them cannot be spelled through it. Refused by name, never mangled. */
function assertAddressable(schema: string): void {
  if (/\b(TEXT|REAL|EPOCH|VECTOR_TYPE|LABEL_INDEX_METHOD)\b|instr\(|[?'"]/.test(schema)) {
    throw new Error(`migrate: schema name ${JSON.stringify(schema)} cannot be read through the SQL renderer`);
  }
}

async function openSource(connectionString: string): Promise<PostgresDriver> {
  // No schema on the target: opening must not CREATE one for a graph that was
  // never on Postgres. Tables are addressed schema-qualified instead.
  return PostgresDriver.open({ driver: 'postgres', connectionString });
}

async function hasGraphTables(src: PostgresDriver, schema: string): Promise<boolean> {
  const rows = await src.all(
    `SELECT COUNT(*) AS n FROM information_schema.tables WHERE table_schema = ? AND table_name IN ('meta', 'nodes', 'node_versions', 'edges', 'access', 'node_vectors')`,
    [schema],
  );
  return num(rows[0]?.n ?? 0) === TABLES.length + 1;
}

async function sourceCounts(src: PostgresDriver, schema: string): Promise<Record<TableName, number>> {
  const out = {} as Record<TableName, number>;
  for (const t of TABLES) out[t] = num((await src.all(`SELECT COUNT(*) AS n FROM ${quoteIdent(schema)}.${t}`))[0]?.n ?? 0);
  return out;
}

/** One row as the SQLite insert binds it. Vectors become int8 BLOBs of the
 *  row's own dims; everything else is already a string, number or null (the
 *  Postgres driver turns int8/numeric into numbers). */
function bindRow(table: TableName, row: SqlRow): SqlParam[] {
  return COLUMNS[table].map((c) => {
    const v = row[c];
    if (table === 'node_vectors' && c === 'embedding') {
      const dims = num(row.dims);
      const values = decodeVector(v, dims);
      if (values.length !== dims) throw new Error(`vector of node '${String(row.id)}' has ${values.length} values but dims ${dims}`);
      return encodeInt8(values);
    }
    if (v === undefined) return null;
    if (typeof v === 'bigint') return Number(v);
    return v as SqlParam;
  });
}

/**
 * Copy the graph held in Postgres schema `schema` into the SQLite file `file`.
 * Refuses when `file` already exists (a file is never overwritten). Throws on
 * any failure, leaving no file behind; the source is untouched either way.
 */
export async function migrateSchemaToSqlite(connectionString: string, schema: string, file: string): Promise<MigrationResult> {
  if (existsSync(file)) throw new Error(`migrate: ${file} already exists — refusing to overwrite it`);
  assertAddressable(schema);
  const started = Date.now();
  const src = await openSource(connectionString);
  const tmp = `${file}.migrating`;
  const removeTmp = (): void => { for (const s of ['', '-wal', '-shm', '-journal']) rmSync(tmp + s, { force: true }); };
  let dst: SqliteDriver | undefined;
  try {
    if (!(await hasGraphTables(src, schema))) return { status: 'absent', schema };
    return await src.transaction(async () => {
      await src.exec('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY');
      const counts = await sourceCounts(src, schema);
      if (TABLES.every((t) => counts[t] === 0)) return { status: 'empty', schema } as MigrationResult;

      removeTmp(); // a temp file left by a crashed attempt is incomplete by definition
      dst = new SqliteDriver(tmp);
      await initSchema(dst);
      const target = dst;
      await target.transaction(async () => {
        for (const m of await src.all(`SELECT key, value FROM ${quoteIdent(schema)}.meta ORDER BY key`)) {
          await target.run('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)', [m.key as string, m.value as string | null]);
        }
        await target.run('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)', [MIGRATED_FROM, schema]);
        for (const t of TABLES) {
          const cols = COLUMNS[t];
          const pk = PRIMARY_KEY[t];
          // A pgvector column arrives as its text form `[…]`, a BYTEA one as
          // bytes; decodeVector reads both.
          const select = cols.join(', ');
          const insert = `INSERT INTO ${t} (${cols.join(', ')}) VALUES (${cols.map(() => '?').join(', ')})`;
          // Keyset paging in primary-key order: each page is one index range.
          let last: SqlParam[] | undefined;
          for (;;) {
            const where = last ? ` WHERE (${pk.join(', ')}) > (${pk.map(() => '?').join(', ')})` : '';
            const rows = await src.all(
              `SELECT ${select} FROM ${quoteIdent(schema)}.${t}${where} ORDER BY ${pk.join(', ')} LIMIT ${PAGE}`,
              last ?? [],
            );
            for (const r of rows) await target.run(insert, bindRow(t, r));
            if (rows.length < PAGE) break;
            const end = rows[rows.length - 1];
            last = pk.map((k) => end[k] as SqlParam);
          }
        }
      });
      for (const t of TABLES) {
        const got = num((await target.all(`SELECT COUNT(*) AS n FROM ${t}`))[0]?.n ?? 0);
        if (got !== counts[t]) throw new Error(`table ${t}: copied ${got} rows, the source has ${counts[t]}`);
      }
      // Fold the WAL into the file so the renamed file is complete on its own;
      // the driver turns WAL on again when the graph is opened.
      await target.exec('PRAGMA journal_mode=DELETE');
      target.close();
      dst = undefined;
      renameSync(tmp, file);
      return { status: 'migrated', schema, counts, ms: Date.now() - started } as MigrationResult;
    });
  } catch (err) {
    if (dst) dst.close();
    removeTmp();
    throw err;
  } finally {
    await src.close().catch(() => {});
  }
}

/**
 * Legacy schemas that still hold rows and that no `*.sqlite` file in `dir`
 * records as its source (`meta.migrated_from`). Empty schemas are not listed:
 * there is nothing in them to lose. A server can stop needing the Postgres
 * server once this is empty.
 */
export async function unmigratedSchemas(connectionString: string, dir: string): Promise<string[]> {
  const taken = new Set<string>();
  if (existsSync(dir)) {
    for (const name of readdirSync(dir)) {
      if (!name.endsWith('.sqlite')) continue;
      const db = new DatabaseSync(join(dir, name), { readOnly: true });
      try {
        const row = db.prepare('SELECT value FROM meta WHERE key = ?').get(MIGRATED_FROM) as { value?: string } | undefined;
        if (row?.value) taken.add(row.value);
      } catch {
        // A file without a meta table is not a graph this engine wrote.
      } finally {
        db.close();
      }
    }
  }
  const src = await openSource(connectionString);
  try {
    const schemas = (await src.all(`SELECT DISTINCT table_schema AS s FROM information_schema.tables WHERE table_name = 'nodes' ORDER BY 1`)).map((r) => r.s as string);
    const out: string[] = [];
    for (const s of schemas) {
      if (taken.has(s)) continue;
      try { assertAddressable(s); } catch { out.push(s); continue; } // unreadable here = still holds what it holds
      if (!(await hasGraphTables(src, s))) continue;
      const counts = await sourceCounts(src, s);
      if (TABLES.some((t) => counts[t] > 0)) out.push(s);
    }
    return out;
  } finally {
    await src.close().catch(() => {});
  }
}

/**
 * Leave an empty graph file at `file` that records `schema` as taken over —
 * what a server moving off Postgres writes when a graph is dropped, so the
 * dropped graph is not copied back from its old schema on the next open.
 */
export async function markDropped(file: string, schema: string): Promise<void> {
  const driver = new SqliteDriver(file);
  try {
    await initSchema(driver);
    await driver.run('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)', [MIGRATED_FROM, schema]);
  } finally {
    driver.close();
  }
}
