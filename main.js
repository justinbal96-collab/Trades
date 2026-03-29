import { createRoot } from "react-dom/client";

import { App } from "./App.js";
import { html } from "./lib.js";
import { HashRouter } from "./router.js";
import { useDashboardData } from "./hooks/useDashboardData.js";

function Root() {
  const { data, loading, error, refresh } = useDashboardData(30000);
  return html`<${HashRouter}><${App} data=${data} loading=${loading} error=${error} refresh=${refresh} /><//>`;
}

const root = createRoot(document.getElementById("root"));
root.render(html`<${Root} />`);
