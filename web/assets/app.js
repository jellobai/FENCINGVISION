const form = document.getElementById("analyze-form");
const statusBox = document.getElementById("status");
const resultsSection = document.getElementById("results");
const submitButton = document.getElementById("submit-button");
const summaryNode = document.getElementById("summary");
const metricsNode = document.getElementById("metrics");
const analysisMetaNode = document.getElementById("analysis-meta");
const focusNode = document.getElementById("practice-focus");
const phrasesNode = document.getElementById("phrases");
const distanceChartNode = document.getElementById("distance-chart");
const stripChartNode = document.getElementById("strip-chart");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  submitButton.disabled = true;
  setStatus("Uploading and analyzing your bout...");
  resultsSection.classList.add("hidden");

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Analysis failed.");
    }

    const data = await response.json();
    renderResults(data);
    setStatus("Analysis complete.");
  } catch (error) {
    setStatus("The upload or analysis step failed. Please try again.");
    console.error(error);
  } finally {
    submitButton.disabled = false;
  }
});

function setStatus(message) {
  statusBox.textContent = message;
  statusBox.classList.remove("hidden");
}

function renderResults(data) {
  summaryNode.textContent = data.summary;
  metricsNode.innerHTML = "";
  analysisMetaNode.innerHTML = "";
  focusNode.innerHTML = "";
  phrasesNode.innerHTML = "";
  distanceChartNode.innerHTML = "";
  stripChartNode.innerHTML = "";

  const metricLabels = [
    ["phrases_detected", "Phrases detected"],
    ["tempo_bursts", "Tempo bursts"],
    ["near_simultaneous_phrases", "Near-simultaneous"],
    ["delayed_commitment_flags", "Delayed commitment"],
  ];

  metricLabels.forEach(([key, label]) => {
    metricsNode.appendChild(buildMetricCard(data.metrics[key], label));
  });

  [
    [data.analysis_meta.detection_source, "Detection source"],
    [data.analysis_meta.tracking_mode, "Tracking mode"],
    [`${data.analysis_meta.duration_seconds}s`, "Duration"],
    [data.analysis_meta.fps, "FPS"],
    [data.analysis_meta.resolution, "Resolution"],
  ].forEach(([value, label]) => {
    analysisMetaNode.appendChild(buildMetricCard(value, label));
  });

  data.practice_focus.forEach((item) => {
    const card = document.createElement("article");
    card.className = "focus-card";
    card.innerHTML = `
      <h3>${item.title}</h3>
      <p>${item.reason}</p>
      <p><strong>Practice idea:</strong> ${item.practice_idea}</p>
    `;
    focusNode.appendChild(card);
  });

  data.phrases.forEach((phrase) => {
    const card = document.createElement("article");
    card.className = "phrase-card";
    card.innerHTML = `
      <h3>Phrase ${phrase.phrase_id}</h3>
      <p>${phrase.start_time} to ${phrase.end_time}</p>
      <p>Moved first: <strong>${phrase.moved_first}</strong></p>
      <p>Committed first: <strong>${phrase.committed_first}</strong></p>
      <p>Pressure owner: <strong>${phrase.pressure_owner}</strong></p>
      <p>${phrase.distance_trend}, ${phrase.tempo_label}, ${phrase.strip_label}</p>
      <p>Near-simultaneous: <strong>${phrase.near_simultaneous}</strong></p>
      <p>Delayed commitment: <strong>${phrase.delayed_commitment}</strong></p>
    `;
    phrasesNode.appendChild(card);
  });

  renderSparkChart(
    distanceChartNode,
    data.charts.distance.slice(0, 32).map((point) => ({
      label: `${point.time_seconds.toFixed(1)}s`,
      primary: normalize(point.distance_pixels, data.charts.distance.map((row) => row.distance_pixels)),
      secondary: point.tempo_burst ? 100 : 0,
    })),
    "Distance"
  );

  renderSparkChart(
    stripChartNode,
    data.charts.strip.slice(0, 32).map((point) => ({
      label: `${point.time_seconds.toFixed(1)}s`,
      primary: normalize(point.left_center_x, data.charts.strip.map((row) => row.left_center_x)),
      secondary: normalize(point.right_center_x, data.charts.strip.map((row) => row.right_center_x)),
    })),
    "Position"
  );

  resultsSection.classList.remove("hidden");
}

function renderSparkChart(container, rows, labelPrefix) {
  rows.forEach((row) => {
    const wrapper = document.createElement("div");
    wrapper.className = "spark-row";
    wrapper.innerHTML = `
      <div class="spark-label">${labelPrefix} ${row.label}</div>
      <div class="spark-line">
        <span class="spark-dot" style="left:${row.primary}%"></span>
        <span class="spark-dot alt" style="left:${row.secondary}%"></span>
      </div>
    `;
    container.appendChild(wrapper);
  });
}

function buildMetricCard(value, label) {
  const card = document.createElement("div");
  card.className = "metric-card";
  card.innerHTML = `<strong>${value}</strong><span>${label}</span>`;
  return card;
}

function normalize(value, allValues) {
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  if (max === min) {
    return 50;
  }
  return ((value - min) / (max - min)) * 100;
}
