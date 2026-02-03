import { applyStep, createBoard, resetBoard } from "./board.js";
import { renderMetrics } from "./metrics.js";

const boardEl = document.getElementById("board");
const metricsEl = document.getElementById("metrics");
const boardMetaEl = document.getElementById("board-meta");
const replaySelect = document.getElementById("replay-select");
const statusLine = document.getElementById("status-line");
const stepCounter = document.getElementById("step-counter");
const playBtn = document.getElementById("play-btn");
const pauseBtn = document.getElementById("pause-btn");
const nextBtn = document.getElementById("next-btn");
const prevBtn = document.getElementById("prev-btn");
const resetBtn = document.getElementById("reset-btn");

let board = null;
let replays = [];
let currentReplayIndex = 0;
let currentStepIndex = 0;
let timer = null;

async function fetchJson(path, allowMissing = false) {
  const response = await fetch(path);
  if (!response.ok) {
    if (allowMissing) {
      return null;
    }
    throw new Error(`Failed to load ${path}`);
  }
  return response.json();
}

async function loadConfig() {
  const config = await fetchJson("data/config.json", true);
  if (!config) {
    boardMetaEl.textContent = "Config not found.";
    return { height: 10, width: 10 };
  }
  const height = config.H ?? config.height ?? 10;
  const width = config.W ?? config.width ?? 10;
  const ships = config.ships ?? [];
  boardMetaEl.textContent = `Board ${height}x${width} | Ships: ${ships.join(", ")}`;
  return { height, width };
}

async function loadMetrics() {
  const metrics = await fetchJson("data/metrics.json", true);
  renderMetrics(metricsEl, metrics);
}

async function loadReplays(maxReplays = 10) {
  const loaded = [];
  for (let idx = 0; idx < maxReplays; idx += 1) {
    const replay = await fetchJson(`data/replays/replay_${idx}.json`, true);
    if (!replay) {
      break;
    }
    loaded.push(replay);
  }
  return loaded;
}

function updateStatus() {
  const replay = replays[currentReplayIndex];
  const totalSteps = replay?.steps?.length ?? 0;
  const stepLabel = `Step ${currentStepIndex} / ${totalSteps}`;
  stepCounter.textContent = stepLabel;
  if (currentStepIndex === 0) {
    statusLine.textContent = "Replay reset.";
    return;
  }
  const lastStep = replay.steps[currentStepIndex - 1];
  statusLine.textContent = `Last: ${lastStep.type} at (${lastStep.r}, ${lastStep.c})`;
}

function applyStepsThrough(index) {
  resetBoard(board);
  const replay = replays[currentReplayIndex];
  for (let i = 0; i < index; i += 1) {
    applyStep(board, replay.steps[i]);
  }
  currentStepIndex = index;
  updateStatus();
}

function stepNext() {
  const replay = replays[currentReplayIndex];
  if (!replay || currentStepIndex >= replay.steps.length) {
    return;
  }
  applyStep(board, replay.steps[currentStepIndex]);
  currentStepIndex += 1;
  updateStatus();
}

function stepPrev() {
  if (currentStepIndex <= 1) {
    applyStepsThrough(0);
    return;
  }
  applyStepsThrough(currentStepIndex - 1);
}

function play() {
  if (timer) {
    return;
  }
  timer = setInterval(() => {
    const replay = replays[currentReplayIndex];
    if (!replay || currentStepIndex >= replay.steps.length) {
      pause();
      return;
    }
    stepNext();
  }, 500);
}

function pause() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

function bindControls() {
  replaySelect.addEventListener("change", (event) => {
    currentReplayIndex = Number(event.target.value);
    applyStepsThrough(0);
  });
  playBtn.addEventListener("click", play);
  pauseBtn.addEventListener("click", pause);
  nextBtn.addEventListener("click", stepNext);
  prevBtn.addEventListener("click", stepPrev);
  resetBtn.addEventListener("click", () => applyStepsThrough(0));
}

async function init() {
  const { height, width } = await loadConfig();
  board = createBoard(boardEl, height, width);
  await loadMetrics();
  replays = await loadReplays();
  replaySelect.innerHTML = "";
  if (replays.length === 0) {
    statusLine.textContent = "No replays found.";
    return;
  }
  replays.forEach((replay, idx) => {
    const option = document.createElement("option");
    option.value = String(idx);
    option.textContent = `Seed ${replay.seed ?? idx}`;
    replaySelect.appendChild(option);
  });
  bindControls();
  applyStepsThrough(0);
}

init();
