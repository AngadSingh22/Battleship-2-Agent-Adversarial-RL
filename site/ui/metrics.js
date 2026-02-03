export function renderMetrics(container, metrics) {
  container.innerHTML = "";
  if (!metrics) {
    container.textContent = "Metrics unavailable.";
    return;
  }

  const rows = [
    { label: "Mean Shots", value: metrics.mean?.toFixed(2) ?? "n/a" },
    { label: "Std Dev", value: metrics.std?.toFixed(2) ?? "n/a" },
    { label: "90th Percentile", value: metrics.p90?.toFixed(2) ?? "n/a" },
    { label: "Generalization Gap", value: metrics.gap?.toFixed(2) ?? "n/a" },
  ];

  rows.forEach((row) => {
    const div = document.createElement("div");
    div.className = "metric-row";
    div.innerHTML = `<span>${row.label}</span><strong>${row.value}</strong>`;
    container.appendChild(div);
  });
}
