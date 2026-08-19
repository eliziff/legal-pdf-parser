import assert from 'node:assert/strict';
import test from 'node:test';
import { orderLayoutLines } from './layout-order.js';

test('moves paired column footnotes after both body columns', () => {
  const line = (id, x0, y0, height = 20) => ({ id, x0, x1: x0 + 360, y0, y1: y0 + height });
  const leftBody = Array.from({ length: 12 }, (_, i) => line(`lb${i}`, 75, 120 + i * 25));
  const leftNotes = Array.from({ length: 4 }, (_, i) => line(`lf${i}`, 92, 500 + i * 20, 16));
  const rightBody = Array.from({ length: 12 }, (_, i) => line(`rb${i}`, 465, 120 + i * 25));
  const rightNotes = Array.from({ length: 4 }, (_, i) => line(`rf${i}`, 480, 500 + i * 20, 16));
  const ordered = orderLayoutLines([...leftBody, ...leftNotes, ...rightBody, ...rightNotes], 900, 700);
  assert.deepEqual(ordered.map(item => item.id), [...leftBody, ...rightBody, ...leftNotes, ...rightNotes].map(item => item.id));
});

test('leaves ordinary two-column pages untouched', () => {
  const lines = Array.from({ length: 20 }, (_, i) => ({ x0: i < 10 ? 75 : 465, x1: i < 10 ? 435 : 825, y0: 100 + (i % 10) * 25, y1: 120 + (i % 10) * 25 }));
  assert.deepEqual(orderLayoutLines(lines, 900, 700), lines);
});
