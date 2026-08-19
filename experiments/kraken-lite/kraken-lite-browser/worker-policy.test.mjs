import test from 'node:test';
import assert from 'node:assert/strict';
import {recognitionWorkers} from './worker-policy.js';

test('recognition workers follow the browser CPU budget and reserve the UI',()=>{
  assert.deepEqual([1,2,4,8,16].map(recognitionWorkers),[1,1,3,7,15]);
});
