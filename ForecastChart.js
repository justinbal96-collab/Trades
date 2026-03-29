import { html, useMemo } from "../lib.js";

function makePath(values, width, height, padding = 12) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const xStep = (width - padding * 2) / Math.max(values.length - 1, 1);

  const points = values.map((value, index) => {
    const x = padding + index * xStep;
    const y =
      height -
      padding -
      ((value - min) / Math.max(max - min, 1e-9)) * (height - padding * 2);
    return [x, y];
  });

  const line = points
    .map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(" ");
  const area = `${line} L ${padding + (values.length - 1) * xStep} ${height - padding} L ${padding} ${height - padding} Z`;

  return { line, area };
}

export function ForecastChart({ forecast = {} }) {
  const points = forecast.points?.length ? forecast.points : [1, 1, 1, 1];
  const path = useMemo(() => makePath(points, 880, 220), [points]);

  return html`
    <section className="panel forecast-panel">
      <div className="section-head">
        <h3>NQ Forecast Path</h3>
        <span>${forecast.horizon_label || "5m horizon • cost-adjusted"}</span>
      </div>
      <svg id="forecast-chart" viewBox="0 0 880 220" preserveAspectRatio="none">
        <defs>
          <linearGradient id="lineGrad" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#39f8ba" stopOpacity="0.75" />
            <stop offset="100%" stopColor="#39f8ba" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <path className="area-path" d=${path.area}></path>
        <path className="line-glow" d=${path.line}></path>
        <path className="line-path" d=${path.line}></path>
      </svg>
    </section>
  `;
}
