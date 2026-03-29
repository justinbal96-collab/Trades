function resolveDashboardApiUrl() {
  const configured = globalThis.__NQ_DASHBOARD_API_URL__;
  if (typeof configured === "string" && configured.trim()) {
    return configured.trim();
  }
  return "/api/dashboard";
}

function resolveFallbackPath() {
  return "./dashboard-fallback.json";
}

export async function fetchDashboardData() {
  const apiUrl = resolveDashboardApiUrl();
  try {
    const response = await fetch(apiUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Dashboard API failed (${response.status})`);
    }
    return response.json();
  } catch (error) {
    const isGithubPages = globalThis.location?.hostname?.includes("github.io");
    if (!isGithubPages) {
      throw error;
    }

    const fallbackResponse = await fetch(resolveFallbackPath(), { cache: "no-store" });
    if (!fallbackResponse.ok) {
      throw new Error(
        `Dashboard API failed and fallback is missing (${fallbackResponse.status}). Set window.__NQ_DASHBOARD_API_URL__ in site-config.js to your live backend /api/dashboard URL.`,
      );
    }
    return fallbackResponse.json();
  }
}
