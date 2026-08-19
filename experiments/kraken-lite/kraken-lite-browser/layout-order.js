const median = values => {
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] || 0;
};

function footerStart(lines, pageHeight) {
  const sorted = lines.slice().sort((a, b) => a.y0 - b.y0);
  if (sorted.length < 8) return null;
  const bodyHeight = median(sorted.slice(0, Math.ceil(sorted.length * .65)).map(line => line.y1 - line.y0));
  for (let index = Math.ceil(sorted.length * .45); index < sorted.length - 2; index += 1) {
    const gap = sorted[index].y0 - sorted[index - 1].y1;
    const tailHeight = median(sorted.slice(index).map(line => line.y1 - line.y0));
    if (sorted[index].y0 > pageHeight * .55 && gap >= Math.max(12, bodyHeight * .75) && tailHeight <= bodyHeight * .92) {
      return sorted[index].y0;
    }
  }
  return null;
}

// Tesseract normally returns useful reading order, except that multi-column
// footnotes are attached to each column. Move paired bottom footnote zones
// after both body columns while otherwise preserving Tesseract's order.
export function orderLayoutLines(lines, pageWidth, pageHeight) {
  if (lines.length < 16) return lines;
  const left = lines.filter(line => (line.x0 + line.x1) / 2 < pageWidth * .48);
  const right = lines.filter(line => (line.x0 + line.x1) / 2 > pageWidth * .52);
  if (left.length < 8 || right.length < 8) return lines;
  const leftStart = footerStart(left, pageHeight), rightStart = footerStart(right, pageHeight);
  if (leftStart === null || rightStart === null || Math.abs(leftStart - rightStart) > pageHeight * .08) return lines;
  const isFooter = line => {
    const center = (line.x0 + line.x1) / 2;
    return center < pageWidth * .48 ? line.y0 >= leftStart : center > pageWidth * .52 && line.y0 >= rightStart;
  };
  const body = lines.filter(line => !isFooter(line));
  const footers = lines.filter(isFooter).sort((a, b) => {
    const column = Number((a.x0 + a.x1) / 2 > pageWidth / 2) - Number((b.x0 + b.x1) / 2 > pageWidth / 2);
    return column || a.y0 - b.y0 || a.x0 - b.x0;
  });
  return [...body, ...footers];
}
