import { useEffect, useRef } from "react";
import type { SpectrogramData } from "../api";
import { magmaColor } from "../spectrogram";

export function SpectrogramCanvas({ data }: { data: SpectrogramData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = data.width;
    canvas.height = data.height;
    const context = canvas.getContext("2d");
    if (!context) return;
    const pixels = context.createImageData(data.width, data.height);
    for (let row = 0; row < data.height; row += 1) {
      for (let column = 0; column < data.width; column += 1) {
        const [red, green, blue] = magmaColor(
          data.values[data.height - row - 1][column],
        );
        const offset = (row * data.width + column) * 4;
        pixels.data.set([red, green, blue, 255], offset);
      }
    }
    context.putImageData(pixels, 0, 0);
  }, [data]);

  return <canvas className="source-canvas" ref={canvasRef} />;
}
