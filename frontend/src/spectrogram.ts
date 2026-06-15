export const VIEWPORT_COLUMNS = 128;

export function createRollingBuffer(
  height: number,
  width = VIEWPORT_COLUMNS,
): Float32Array {
  return new Float32Array(height * width);
}

export function appendSpectrogramColumn(
  buffer: Float32Array,
  height: number,
  width: number,
  column: number[],
): void {
  if (column.length !== height || buffer.length !== height * width) {
    throw new Error("Spectrogram buffer dimensions do not match");
  }
  for (let row = 0; row < height; row += 1) {
    const offset = row * width;
    buffer.copyWithin(offset, offset + 1, offset + width);
    buffer[offset + width - 1] = column[row];
  }
}

export function currentActivity(column: number[]): number {
  if (column.length === 0) return 0;
  const energy = column.reduce((sum, value) => sum + value * value, 0);
  return Math.sqrt(energy / column.length);
}

export function magmaColor(value: number): [number, number, number] {
  const stops: Array<[number, number, number, number]> = [
    [0, 4, 4, 18],
    [0.18, 45, 17, 88],
    [0.38, 108, 27, 113],
    [0.58, 181, 54, 121],
    [0.78, 247, 125, 83],
    [1, 252, 253, 191],
  ];
  const clamped = Math.max(0, Math.min(1, value));
  const upperIndex = stops.findIndex(([position]) => position >= clamped);
  if (upperIndex <= 0) return stops[0].slice(1) as [number, number, number];
  const lower = stops[upperIndex - 1];
  const upper = stops[upperIndex];
  const ratio = (clamped - lower[0]) / (upper[0] - lower[0]);
  return [
    Math.round(lower[1] + (upper[1] - lower[1]) * ratio),
    Math.round(lower[2] + (upper[2] - lower[2]) * ratio),
    Math.round(lower[3] + (upper[3] - lower[3]) * ratio),
  ];
}
