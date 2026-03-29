import { fetchDashboardData } from "../api.js";
import { useEffect, useState } from "../lib.js";

export function useDashboardData(refreshMs = 30000) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    let mounted = true;

    async function load() {
      try {
        const payload = await fetchDashboardData();
        if (!mounted) return;
        setData(payload);
        setError(null);
      } catch (err) {
        if (!mounted) return;
        setError(err.message || "Unable to fetch dashboard data");
      } finally {
        if (mounted) setLoading(false);
      }
    }

    load();
    const id = setInterval(load, refreshMs);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [refreshMs, refreshToken]);

  return {
    data,
    loading,
    error,
    refresh: () => setRefreshToken((x) => x + 1),
  };
}
