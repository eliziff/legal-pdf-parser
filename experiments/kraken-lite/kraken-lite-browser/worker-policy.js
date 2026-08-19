export function recognitionWorkers(parallelism) {
  const available=Math.max(1,Math.floor(Number(parallelism)||1));
  return available>2?available-1:1;
}
