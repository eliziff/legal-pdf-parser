import assert from 'node:assert/strict';
import test from 'node:test';
import { preprocessLine } from './line-preprocess.js';

const rgba = values => Uint8ClampedArray.from(values.flatMap(value => [value, value, value, 255]));

test('grayscale normalization maps white to zero ink and black to one', () => {
  assert.deepEqual([...preprocessLine(rgba([255, 0]), 2, 1)], [0, 1]);
});

test('Otsu and Sauvola preserve a dark stroke on light paper', () => {
  const pixels = rgba([240, 235, 20, 230, 245, 240]);
  for (const mode of ['otsu', 'sauvola']) {
    const output = preprocessLine(pixels, 6, 1, mode);
    assert.equal(output[2], 1);
    assert.equal(output[0], 0);
  }
});
