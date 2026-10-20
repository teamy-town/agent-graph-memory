/**
 * Work for a caller that has gone is not run (driver.ts withSignal): a
 * request queued behind another on a graph's one connection is dropped when
 * its caller leaves, instead of holding the connection for an answer nobody
 * reads. A transaction that has begun is never cut.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { CallerGoneError, Serializer, withSignal } from '../src/driver.js';

test('queued work whose caller left is never run, and the queue moves on', async () => {
  const gate = new Serializer();
  let releaseFirst!: () => void;
  const first = gate.exclusive(() => new Promise<string>((r) => { releaseFirst = () => r('first'); }));
  const caller = new AbortController();
  let ran = false;
  const second = withSignal(caller.signal, () => gate.exclusive(async () => { ran = true; return 'second'; }));
  const third = gate.exclusive(async () => 'third');
  caller.abort();
  await new Promise((r) => setImmediate(r)); // the first holds the connection now
  releaseFirst();
  assert.equal(await first, 'first');
  await assert.rejects(second, CallerGoneError);
  assert.equal(ran, false, 'the abandoned statement must not reach the connection');
  assert.equal(await third, 'third');
});

test('a request that is aborted between statements stops at its next statement', async () => {
  const gate = new Serializer();
  const caller = new AbortController();
  const ran: number[] = [];
  const request = withSignal(caller.signal, async () => {
    for (let i = 0; i < 5; i++) {
      await gate.exclusive(async () => { ran.push(i); });
      if (i === 1) caller.abort();
    }
  });
  await assert.rejects(request, CallerGoneError);
  assert.deepEqual(ran, [0, 1]);
});

test('a transaction that has begun runs to the end even when its caller leaves', async () => {
  const gate = new Serializer();
  const caller = new AbortController();
  const log: string[] = [];
  const out = await withSignal(caller.signal, () => gate.transaction(async () => {
    log.push('a');
    caller.abort();
    await gate.exclusive(async () => { log.push('b'); });
    return 'committed';
  }, () => { log.push('BEGIN'); }, () => { log.push('COMMIT'); }, () => { log.push('ROLLBACK'); }));
  assert.equal(out, 'committed');
  assert.deepEqual(log, ['BEGIN', 'a', 'b', 'COMMIT']);
});
