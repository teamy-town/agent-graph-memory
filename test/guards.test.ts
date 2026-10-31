import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkContent } from '../src/guards.js';

const long = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890';

test('a word ending in sk followed by a long id is not an openai-style key', () => {
  for (const text of [`task-${long}`, `risk-${long}`, `disk-${long}`, `{"ref":"agent/task-${long}"}`]) {
    assert.equal(checkContent(text).rejected, false, text);
  }
});

test('an openai-style key is still refused wherever it sits', () => {
  for (const text of [`sk-${long}`, `key sk-${long}`, `{"k":"sk-proj-${long}"}`, `KEY=sk-${long}`, `/sk-${long}`]) {
    const report = checkContent(text);
    assert.equal(report.rejected, true, text);
    assert.match(report.reason ?? '', /openai-style key/);
  }
});
