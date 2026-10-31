import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkContent, guardFields } from '../src/guards.js';

const long = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890';

test('a word ending in sk followed by a long id is not an openai-style key', () => {
  for (const text of [`task-${long}`, `risk-${long}`, `disk-${long}`, `{"ref":"agent/task-${long}"}`]) {
    assert.equal(checkContent(text).rejected, false, text);
  }
});

test('a slug of ordinary words after sk- is not an openai-style key', () => {
  for (const text of ['51-query-sk-between-k-and-k-1-l-1', 'sk-based-sort-key-for-the-thread-index', '{"what":"sk-range-on-the-partition-key-only"}']) {
    assert.equal(checkContent(text).rejected, false, text);
  }
});

test('an openai-style key is still refused wherever it sits', () => {
  for (const text of [`sk-${long}`, `key sk-${long}`, `{"k":"sk-proj-${long}"}`, `KEY=sk-${long}`, `/sk-${long}`, `sk-${'aB3dE5gH7j'.repeat(4)}`, `sk-ant-api03-${'Ab12Cd34Ef'.repeat(3)}-Qr78St90`]) {
    const report = checkContent(text);
    assert.equal(report.rejected, true, text);
    assert.match(report.reason ?? '', /openai-style key/);
  }
});

test('a refused attribute is named by its path, and its value stays out of the message', () => {
  const key = `sk-${long}`;
  assert.throws(() => guardFields('link', { attrs: { exec: 'x', limits: ['fine', { text: key }] } }), (err: Error) => {
    assert.match(err.message, /attrs\.limits\.1\.text looks like a credential/);
    assert.ok(!err.message.includes(key));
    return true;
  });
});
