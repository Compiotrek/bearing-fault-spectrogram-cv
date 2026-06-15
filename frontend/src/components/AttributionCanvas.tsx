import { useEffect, useRef } from "react";
import type { GradCamData, SpectrogramData } from "../api";
import { magmaColor } from "../spectrogram";

export type AttributionMode = "topology" | "focus" | "overlay";

interface Props {
  gradcam: GradCamData;
  spectrogram: SpectrogramData;
  mode: AttributionMode;
}

function attentionColor(value: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, value));
  if (clamped < 0.45) {
    const ratio = clamped / 0.45;
    return [20, Math.round(80 + 120 * ratio), Math.round(140 + 100 * ratio)];
  }
  const ratio = (clamped - 0.45) / 0.55;
  return [Math.round(20 + 235 * ratio), Math.round(200 - 80 * ratio), 70];
}

function drawContours(
  context: CanvasRenderingContext2D,
  values: number[][],
  left: number,
  top: number,
  cellWidth: number,
  cellHeight: number,
  ratio: number,
) {
  const thresholds = [
    { value: 0.35, color: "rgba(112, 186, 197, 0.62)", width: 0.7 },
    { value: 0.58, color: "rgba(197, 139, 69, 0.86)", width: 1.0 },
    { value: 0.78, color: "rgba(202, 101, 72, 0.95)", width: 1.4 },
  ];

  for (const threshold of thresholds) {
    context.beginPath();
    context.strokeStyle = threshold.color;
    context.lineWidth = threshold.width * ratio;
    for (let row = 0; row < values.length; row += 1) {
      for (let column = 0; column < values[row].length; column += 1) {
        if (values[row][column] < threshold.value) continue;
        const x = left + column * cellWidth;
        const y = top + (values.length - row - 1) * cellHeight;
        const neighbors = [
          row === 0 || values[row - 1][column] < threshold.value,
          column === values[row].length - 1 ||
            values[row][column + 1] < threshold.value,
          row === values.length - 1 ||
            values[row + 1][column] < threshold.value,
          column === 0 || values[row][column - 1] < threshold.value,
        ];
        if (neighbors[0]) {
          context.moveTo(x, y + cellHeight);
          context.lineTo(x + cellWidth, y + cellHeight);
        }
        if (neighbors[1]) {
          context.moveTo(x + cellWidth, y);
          context.lineTo(x + cellWidth, y + cellHeight);
        }
        if (neighbors[2]) {
          context.moveTo(x, y);
          context.lineTo(x + cellWidth, y);
        }
        if (neighbors[3]) {
          context.moveTo(x, y);
          context.lineTo(x, y + cellHeight);
        }
      }
    }
    context.stroke();
  }
}

function drawProfile(
  canvas: HTMLCanvasElement,
  values: number[],
  color: string,
  label: string,
) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(200, Math.floor(rect.width * ratio));
  canvas.height = Math.max(80, Math.floor(rect.height * ratio));
  const context = canvas.getContext("2d");
  if (!context) return;
  const width = canvas.width;
  const height = canvas.height;
  context.fillStyle = "#10171a";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(102, 132, 140, 0.16)";
  context.lineWidth = ratio;
  for (let tick = 1; tick < 4; tick += 1) {
    const y = (height * tick) / 4;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
  const maximum = Math.max(...values, 1e-6);
  const gradient = context.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "rgba(112, 186, 197, 0.18)");
  gradient.addColorStop(1, color);
  context.strokeStyle = gradient;
  context.lineWidth = 1.5 * ratio;
  context.beginPath();
  values.forEach((value, index) => {
    const x = (index / Math.max(1, values.length - 1)) * width;
    const y = height - 9 * ratio - (value / maximum) * (height - 22 * ratio);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
  context.fillStyle = "#74868a";
  context.font = `${9 * ratio}px ui-monospace, monospace`;
  context.fillText(label, 8 * ratio, 13 * ratio);
}

export function AttributionCanvas({ gradcam, spectrogram, mode }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timeProfileRef = useRef<HTMLCanvasElement>(null);
  const frequencyProfileRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(620, Math.floor(rect.width * ratio));
      canvas.height = Math.max(360, Math.floor(rect.height * ratio));
      const context = canvas.getContext("2d");
      if (!context) return;

      const padding = {
        left: 46 * ratio,
        right: 18 * ratio,
        top: 20 * ratio,
        bottom: 30 * ratio,
      };
      const width = canvas.width - padding.left - padding.right;
      const height = canvas.height - padding.top - padding.bottom;
      const cellWidth = width / gradcam.width;
      const cellHeight = height / gradcam.height;
      context.fillStyle = "#0d1417";
      context.fillRect(0, 0, canvas.width, canvas.height);

      for (let row = 0; row < gradcam.height; row += 1) {
        for (let column = 0; column < gradcam.width; column += 1) {
          const source = spectrogram.values[row][column];
          const attention = gradcam.values[row][column];
          const [baseRed, baseGreen, baseBlue] = magmaColor(source);
          let red = baseRed;
          let green = baseGreen;
          let blue = baseBlue;

          if (mode === "topology") {
            const dim = 0.32 + attention * 0.68;
            red *= dim;
            green *= dim;
            blue *= dim;
          } else if (mode === "focus") {
            const focused = attention >= 0.72;
            const dim = focused ? 1 : 0.12;
            red *= dim;
            green *= dim;
            blue *= dim;
            if (focused) {
              const [focusRed, focusGreen, focusBlue] =
                attentionColor(attention);
              red = red * 0.42 + focusRed * 0.58;
              green = green * 0.42 + focusGreen * 0.58;
              blue = blue * 0.42 + focusBlue * 0.58;
            }
          } else {
            const [heatRed, heatGreen, heatBlue] =
              attentionColor(attention);
            const alpha = attention * 0.72;
            red = red * (1 - alpha) + heatRed * alpha;
            green = green * (1 - alpha) + heatGreen * alpha;
            blue = blue * (1 - alpha) + heatBlue * alpha;
          }

          context.fillStyle = `rgb(${red}, ${green}, ${blue})`;
          context.fillRect(
            padding.left + column * cellWidth,
            padding.top + (gradcam.height - row - 1) * cellHeight,
            Math.ceil(cellWidth + 0.5),
            Math.ceil(cellHeight + 0.5),
          );
        }
      }

      context.strokeStyle = "rgba(94, 132, 143, 0.17)";
      context.lineWidth = ratio;
      for (let tick = 0; tick <= 8; tick += 1) {
        const x = padding.left + (width * tick) / 8;
        context.beginPath();
        context.moveTo(x, padding.top);
        context.lineTo(x, padding.top + height);
        context.stroke();
      }
      for (let tick = 0; tick <= 4; tick += 1) {
        const y = padding.top + (height * tick) / 4;
        context.beginPath();
        context.moveTo(padding.left, y);
        context.lineTo(padding.left + width, y);
        context.stroke();
      }

      if (mode !== "overlay") {
        drawContours(
          context,
          gradcam.values,
          padding.left,
          padding.top,
          cellWidth,
          cellHeight,
          ratio,
        );
      }

      const peakX =
        padding.left + (gradcam.peak_time_index + 0.5) * cellWidth;
      const peakY =
        padding.top +
        (gradcam.height - gradcam.peak_frequency_index - 0.5) * cellHeight;
      context.strokeStyle = "#e1ddd2";
      context.lineWidth = 1.2 * ratio;
      context.beginPath();
      context.arc(peakX, peakY, 7 * ratio, 0, Math.PI * 2);
      context.moveTo(peakX - 12 * ratio, peakY);
      context.lineTo(peakX + 12 * ratio, peakY);
      context.moveTo(peakX, peakY - 12 * ratio);
      context.lineTo(peakX, peakY + 12 * ratio);
      context.stroke();
      context.fillStyle = "#74868a";
      context.font = `${9 * ratio}px ui-monospace, monospace`;
      context.textAlign = "right";
      context.fillText("6 kHz", padding.left - 8 * ratio, padding.top + 5 * ratio);
      context.fillText(
        "0 Hz",
        padding.left - 8 * ratio,
        padding.top + height,
      );
      context.textAlign = "center";
      context.fillText("TIME →", padding.left + width / 2, canvas.height - 8 * ratio);
      context.fillStyle = "#d8d6cb";
      context.textAlign = "left";
      context.fillText(
        "PEAK EVIDENCE",
        Math.min(peakX + 12 * ratio, canvas.width - 115 * ratio),
        Math.max(peakY - 10 * ratio, 14 * ratio),
      );
    };

    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    draw();
    return () => observer.disconnect();
  }, [gradcam, mode, spectrogram]);

  useEffect(() => {
    const redrawProfiles = () => {
      if (timeProfileRef.current) {
        drawProfile(
          timeProfileRef.current,
          gradcam.time_profile,
          "#c58b45",
          "TIME EVIDENCE",
        );
      }
      if (frequencyProfileRef.current) {
        drawProfile(
          frequencyProfileRef.current,
          gradcam.frequency_profile,
          "#70bac5",
          "FREQUENCY EVIDENCE",
        );
      }
    };
    const observer = new ResizeObserver(redrawProfiles);
    if (timeProfileRef.current) observer.observe(timeProfileRef.current);
    if (frequencyProfileRef.current) {
      observer.observe(frequencyProfileRef.current);
    }
    redrawProfiles();
    return () => observer.disconnect();
  }, [gradcam]);

  return (
    <>
      <canvas className="attribution-canvas" ref={canvasRef} />
      <div className="attribution-profiles">
        <canvas ref={timeProfileRef} />
        <canvas ref={frequencyProfileRef} />
      </div>
    </>
  );
}
