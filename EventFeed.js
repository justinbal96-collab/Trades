import { html } from "../lib.js";

export function EventFeed({ events = [] }) {
  return html`
    <section className="panel events-panel">
      <div className="section-head">
        <h3>Execution Feed</h3>
        <span>Live from quant engine</span>
      </div>
      <ul className="feed-list">
        ${events.map(
          (event, idx) => html`
            <li key=${idx}>
              <b>${event.time}</b>
              <span>${event.message}</span>
            </li>
          `
        )}
      </ul>
    </section>
  `;
}
