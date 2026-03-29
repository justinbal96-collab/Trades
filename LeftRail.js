import { html } from "../lib.js";
import { NavLink } from "../router.js";

function cls({ isActive }) {
  return isActive ? "nav-item active" : "nav-item";
}

export function LeftRail({ watchlist = [] }) {
  return html`
    <aside className="left-rail">
      <section className="rail-section">
        <p className="nav-label">WORKSPACE</p>
        <nav className="nav-stack">
          <${NavLink} className=${cls} to="/">Dashboard<//>
          <${NavLink} className=${cls} to="/signals">Signal Studio<//>
          <${NavLink} className=${cls} to="/risk">Risk Envelope<//>
        </nav>
      </section>

      <section className="rail-section">
        <p className="nav-label">WATCHLIST</p>
        <ul className="watchlist">
          ${watchlist.map(
            (item) => html`
              <li key=${item.symbol}>
                <span>${item.symbol}</span>
                <b className=${item.change_pct >= 0 ? "pos" : "neg"}
                  >${item.change_pct >= 0 ? "+" : ""}${item.change_pct.toFixed(2)}%</b
                >
              </li>
            `
          )}
        </ul>
      </section>
    </aside>
  `;
}
