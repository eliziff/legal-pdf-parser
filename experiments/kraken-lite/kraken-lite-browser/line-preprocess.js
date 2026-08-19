function grayscale(rgba) {
  const gray = new Uint8Array(rgba.length / 4);
  for (let i = 0; i < gray.length; i += 1) {
    const offset = i * 4;
    gray[i] = Math.round((rgba[offset] + rgba[offset + 1] + rgba[offset + 2]) / 3);
  }
  return gray;
}

function otsu(gray) {
  const histogram = new Uint32Array(256);
  for (const value of gray) histogram[value] += 1;
  let sum = 0;
  for (let value = 0; value < 256; value += 1) sum += value * histogram[value];
  let left = 0, leftSum = 0, best = -1, threshold = 127;
  for (let value = 0; value < 256; value += 1) {
    left += histogram[value]; leftSum += value * histogram[value];
    if (!left || left === gray.length) continue;
    const score = (leftSum / left - (sum - leftSum) / (gray.length - left)) ** 2 * left * (gray.length - left);
    if (score > best) { best = score; threshold = value; }
  }
  return threshold;
}

function binary(gray, threshold) {
  return Float32Array.from(gray, value => Number(value <= threshold));
}

function sauvola(gray, width, height) {
  const stride = width + 1, sum = new Float64Array((height + 1) * stride), squares = new Float64Array(sum.length);
  for (let y = 0; y < height; y += 1) {
    let rowSum = 0, rowSquares = 0;
    for (let x = 0; x < width; x += 1) {
      const value = gray[y * width + x]; rowSum += value; rowSquares += value * value;
      const index = (y + 1) * stride + x + 1;
      sum[index] = sum[index - stride] + rowSum; squares[index] = squares[index - stride] + rowSquares;
    }
  }
  const radius = 12, output = new Float32Array(gray.length);
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const x0 = Math.max(0, x - radius), y0 = Math.max(0, y - radius), x1 = Math.min(width, x + radius + 1), y1 = Math.min(height, y + radius + 1);
    const a = y0 * stride + x0, b = y0 * stride + x1, c = y1 * stride + x0, d = y1 * stride + x1, count = (x1 - x0) * (y1 - y0);
    const mean = (sum[d] - sum[b] - sum[c] + sum[a]) / count;
    const variance = Math.max(0, (squares[d] - squares[b] - squares[c] + squares[a]) / count - mean * mean);
    output[y * width + x] = Number(gray[y * width + x] <= mean * (1 + .2 * (Math.sqrt(variance) / 128 - 1)));
  }
  return output;
}

function close(binaryPixels, width, height) {
  const dilated = new Float32Array(binaryPixels.length), output = new Float32Array(binaryPixels.length);
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const index = y * width + x;
    dilated[index] = Number(binaryPixels[index] || (x && binaryPixels[index - 1]) || (x + 1 < width && binaryPixels[index + 1]));
  }
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const index = y * width + x;
    output[index] = Number(dilated[index] && (!x || dilated[index - 1]) && (x + 1 === width || dilated[index + 1]));
  }
  return output;
}

function percentile(histogram, count, fraction) {
  let seen = 0;
  for (let value = 0; value < 256; value += 1) if ((seen += histogram[value]) >= count * fraction) return value;
  return 255;
}

export function preprocessLine(rgba, width, height, mode = 'none') {
  const gray = grayscale(rgba);
  if (mode === 'auto') {
    const histogram = new Uint32Array(256);
    for (const value of gray) histogram[value] += 1;
    const low = percentile(histogram, gray.length, .05), high = percentile(histogram, gray.length, .9);
    mode = high < 225 ? 'sauvola' : high - low < 105 ? 'otsu' : 'none';
  }
  if (mode === 'sauvola') return sauvola(gray, width, height);
  const data = mode === 'otsu' || mode === 'morph' ? binary(gray, otsu(gray)) : Float32Array.from(gray, value => 1 - value / 255);
  return mode === 'morph' ? close(data, width, height) : data;
}
