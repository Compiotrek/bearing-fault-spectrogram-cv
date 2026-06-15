import { useCallback, useEffect, useRef, useState } from "react";
import type { SpectrogramData } from "../api";
import {
  VIEWPORT_COLUMNS,
  appendSpectrogramColumn,
  createRollingBuffer,
  currentActivity,
  magmaColor,
} from "../spectrogram";

export type ReplayStatus =
  | "READY"
  | "BUFFERING"
  | "STREAMING"
  | "PAUSED"
  | "WINDOW COMPLETE";

interface Props {
  data: SpectrogramData;
  sampleRate: number;
  onStatusChange: (status: ReplayStatus) => void;
}

const BASE_DURATION_MS = 6500;
const PLOT_PADDING = { left: 48, right: 18, top: 24, bottom: 30 };
const TRACE_HEIGHT_RATIO = 0.22;

export function LiveReplayConsole({
  data,
  sampleRate,
  onStatusChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bufferRef = useRef(createRollingBuffer(data.height));
  const activityBufferRef = useRef(new Float32Array(VIEWPORT_COLUMNS));
  const eventBufferRef = useRef(new Uint8Array(VIEWPORT_COLUMNS));
  const frameRef = useRef(0);
  const animationRef = useRef<number | null>(null);
  const lastTimestampRef = useRef<number | null>(null);
  const accumulatorRef = useRef(0);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [frameIndex, setFrameIndex] = useState(0);
  const [activity, setActivity] = useState(0);
  const [status, setStatus] = useState<ReplayStatus>("READY");

  const updateStatus = useCallback(
    (next: ReplayStatus) => {
      setStatus(next);
      onStatusChange(next);
    },
    [onStatusChange],
  );

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(640, Math.floor(rect.width * ratio));
    const height = Math.max(330, Math.floor(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#111310";
    context.fillRect(0, 0, width, height);

    const left = PLOT_PADDING.left * ratio;
    const right = width - PLOT_PADDING.right * ratio;
    const top = PLOT_PADDING.top * ratio;
    const bottom = height - PLOT_PADDING.bottom * ratio;
    const plotWidth = right - left;
    const totalPlotHeight = bottom - top;
    const traceHeight = totalPlotHeight * TRACE_HEIGHT_RATIO;
    const traceGap = 18 * ratio;
    const waterfallTop = top + traceHeight + traceGap;
    const plotHeight = bottom - waterfallTop;
    const cellWidth = plotWidth / VIEWPORT_COLUMNS;
    const cellHeight = plotHeight / data.height;
    const buffer = bufferRef.current;

    context.fillStyle = "#171a15";
    context.fillRect(left, top, plotWidth, traceHeight);
    context.strokeStyle = "rgba(207, 196, 168, 0.14)";
    context.lineWidth = ratio;
    context.beginPath();
    context.moveTo(left, top + traceHeight / 2);
    context.lineTo(right, top + traceHeight / 2);
    context.stroke();

    const activityBuffer = activityBufferRef.current;
    context.strokeStyle = "#70bac5";
    context.lineWidth = 1.4 * ratio;
    context.beginPath();
    for (let column = 0; column < VIEWPORT_COLUMNS; column += 1) {
      const x = left + column * cellWidth;
      const y =
        top + traceHeight - activityBuffer[column] * (traceHeight * 0.86);
      if (column === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();

    const eventBuffer = eventBufferRef.current;
    for (let column = 0; column < VIEWPORT_COLUMNS; column += 1) {
      if (!eventBuffer[column]) continue;
      const x = left + column * cellWidth;
      context.fillStyle = "#c58b45";
      context.fillRect(x, top, Math.max(1.5 * ratio, cellWidth), 5 * ratio);
    }

    context.fillStyle = "#11130f";
    context.fillRect(left, waterfallTop, plotWidth, plotHeight);
    for (let column = 0; column < VIEWPORT_COLUMNS; column += 1) {
      const persistence = 0.26 + 0.74 * (column / VIEWPORT_COLUMNS) ** 1.5;
      for (let row = 0; row < data.height; row += 1) {
        const value = buffer[row * VIEWPORT_COLUMNS + column];
        const [red, green, blue] = magmaColor(value);
        context.fillStyle = `rgb(${red * persistence}, ${
          green * persistence
        }, ${blue * persistence})`;
        context.fillRect(
          left + column * cellWidth,
          bottom - (row + 1) * cellHeight,
          Math.ceil(cellWidth + 0.5),
          Math.ceil(cellHeight + 0.5),
        );
      }
    }

    context.strokeStyle = "rgba(94, 132, 143, 0.18)";
    context.lineWidth = ratio;
    for (let tick = 0; tick <= 8; tick += 1) {
      const x = left + (plotWidth * tick) / 8;
      context.beginPath();
      context.moveTo(x, waterfallTop);
      context.lineTo(x, bottom);
      context.stroke();
    }
    for (let tick = 0; tick <= 4; tick += 1) {
      const y = waterfallTop + (plotHeight * tick) / 4;
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(right, y);
      context.stroke();
    }

    const scanX = right - cellWidth * 1.5;
    const gradient = context.createLinearGradient(scanX - 18 * ratio, 0, scanX, 0);
    gradient.addColorStop(0, "rgba(112, 186, 197, 0)");
    gradient.addColorStop(1, "rgba(112, 186, 197, 0.34)");
    context.fillStyle = gradient;
    context.fillRect(
      scanX - 18 * ratio,
      top,
      18 * ratio,
      totalPlotHeight,
    );
    context.strokeStyle = "#7bc3cd";
    context.lineWidth = 1.5 * ratio;
    context.beginPath();
    context.moveTo(scanX, top);
    context.lineTo(scanX, bottom);
    context.stroke();

    context.fillStyle = "#8e8b7e";
    context.font = `${10 * ratio}px ui-monospace, monospace`;
    context.textAlign = "right";
    context.fillText("RMS", left - 8 * ratio, top + 5 * ratio);
    context.fillText("6 kHz", left - 8 * ratio, waterfallTop + 4 * ratio);
    context.fillText(
      "3 kHz",
      left - 8 * ratio,
      waterfallTop + plotHeight / 2,
    );
    context.fillText("0 Hz", left - 8 * ratio, bottom);
    context.textAlign = "center";
    context.fillText("− history", left + plotWidth * 0.16, height - 8 * ratio);
    context.fillStyle = "#7bc3cd";
    context.fillText("scan head", scanX, height - 8 * ratio);

    if (frameRef.current === 0) {
      context.fillStyle = "rgba(17, 19, 16, 0.78)";
      context.fillRect(left, top, plotWidth, totalPlotHeight);
      context.fillStyle = "#d9d0ba";
      context.font = `${15 * ratio}px ui-monospace, monospace`;
      context.textAlign = "center";
      context.fillText(
        "ACQUISITION ARMED",
        left + plotWidth / 2,
        top + totalPlotHeight / 2 - 8 * ratio,
      );
      context.fillStyle = "#777466";
      context.font = `${10 * ratio}px ui-monospace, monospace`;
      context.fillText(
        "press REPLAY to feed the stored window",
        left + plotWidth / 2,
        top + totalPlotHeight / 2 + 15 * ratio,
      );
    }
  }, [data.height]);

  const reset = useCallback(
    (continuePlaying = false) => {
      bufferRef.current = createRollingBuffer(data.height);
      activityBufferRef.current = new Float32Array(VIEWPORT_COLUMNS);
      eventBufferRef.current = new Uint8Array(VIEWPORT_COLUMNS);
      frameRef.current = 0;
      accumulatorRef.current = 0;
      lastTimestampRef.current = null;
      setFrameIndex(0);
      setActivity(0);
      updateStatus(continuePlaying ? "BUFFERING" : "READY");
      setPlaying(continuePlaying);
      draw();
    },
    [data.height, draw, updateStatus],
  );

  useEffect(() => {
    reset(false);
  }, [data.sample_id, reset]);

  useEffect(() => {
    const observer = new ResizeObserver(draw);
    if (canvasRef.current) observer.observe(canvasRef.current);
    draw();
    return () => observer.disconnect();
  }, [draw]);

  useEffect(() => {
    if (!playing) {
      draw();
      return;
    }
    const columnDuration = BASE_DURATION_MS / data.width / speed;

    const animate = (timestamp: number) => {
      if (lastTimestampRef.current === null) lastTimestampRef.current = timestamp;
      const frameDelta = Math.min(timestamp - lastTimestampRef.current, 48);
      accumulatorRef.current += frameDelta;
      lastTimestampRef.current = timestamp;

      let updated = false;
      while (
        accumulatorRef.current >= columnDuration &&
        frameRef.current < data.width
      ) {
        const index = frameRef.current;
        const column = data.values.map((row) => row[index]);
        const nextActivity = currentActivity(column);
        appendSpectrogramColumn(
          bufferRef.current,
          data.height,
          VIEWPORT_COLUMNS,
          column,
        );
        activityBufferRef.current.copyWithin(0, 1);
        activityBufferRef.current[VIEWPORT_COLUMNS - 1] = nextActivity;
        eventBufferRef.current.copyWithin(0, 1);
        eventBufferRef.current[VIEWPORT_COLUMNS - 1] =
          nextActivity > 0.67 ? 1 : 0;
        frameRef.current += 1;
        accumulatorRef.current -= columnDuration;
        setFrameIndex(frameRef.current);
        setActivity(nextActivity);
        updateStatus(
          frameRef.current < VIEWPORT_COLUMNS ? "BUFFERING" : "STREAMING",
        );
        updated = true;
      }
      if (updated) draw();

      if (frameRef.current >= data.width) {
        updateStatus("WINDOW COMPLETE");
        if (loop) {
          reset(true);
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setPlaying(false);
        }
        return;
      }
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [data, draw, loop, playing, reset, speed, updateStatus]);

  const elapsedMs = (frameIndex / data.width) * (2048 / sampleRate) * 1000;

  return (
    <section className="scope-frame">
      <header className="scope-frame__title-rail">
        <div>
          <span className="module-number">Acquisition monitor / CH A</span>
          <h2>Vibration playback</h2>
          <small>rolling time-frequency record</small>
        </div>
        <span className={`status-tag status-tag--${status.toLowerCase().replace(" ", "-")}`}>
          {status}
        </span>
      </header>

      <div className="scope-frame__viewport">
        <span className="scope-corners" aria-hidden="true" />
        <canvas ref={canvasRef} />
        <span className="canvas-channel">CH A · DRIVE-END VIBRATION</span>
        <span className="scope-readout">RMS ENVELOPE · STFT 256 / 128</span>
        <span className={`frame-state ${playing ? "is-active" : ""}`}>
          {playing ? "FRAME INPUT" : "INPUT HOLD"}
        </span>
      </div>

      <div className="scope-frame__metadata-rail">
        <div>
          <span>Replay time</span>
          <strong>{elapsedMs.toFixed(1)} ms</strong>
        </div>
        <div>
          <span>Frame</span>
          <strong>
            {String(frameIndex).padStart(3, "0")} / {data.width}
          </strong>
        </div>
        <div>
          <span>Signal activity</span>
          <strong>{(activity * 100).toFixed(0)}%</strong>
        </div>
        <div>
          <span>Sample rate</span>
          <strong>{sampleRate / 1000} kHz</strong>
        </div>
      </div>

      <div className="scope-frame__controls">
        <button
          className="console-button console-button--primary"
          onClick={() => {
            if (frameIndex >= data.width) {
              reset(true);
            } else {
              if (playing) {
                setPlaying(false);
                updateStatus("PAUSED");
              } else {
                setPlaying(true);
                updateStatus(
                  frameIndex === 0 ? "BUFFERING" : "STREAMING",
                );
              }
            }
          }}
        >
          {playing ? "Pause" : "Replay"}
        </button>
        <button className="console-button" onClick={() => reset(false)}>
          Reset
        </button>
        <div className="console-button-group" aria-label="Playback speed">
          {[0.5, 1, 2].map((value) => (
            <button
              className={`console-button console-button--small ${
                speed === value ? "is-active" : ""
              }`}
              key={value}
              onClick={() => setSpeed(value)}
            >
              {value}×
            </button>
          ))}
        </div>
        <label className="console-toggle">
          <input
            type="checkbox"
            checked={loop}
            onChange={(event) => setLoop(event.target.checked)}
          />
          <span />
          Loop
        </label>
      </div>
    </section>
  );
}
