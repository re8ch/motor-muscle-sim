import { HumanoidMotorSim } from "./humanoid.js";
import { MotorMuscleSim, parseGrid } from "./simulator.js";
import { FieldRenderer, HumanoidRenderer, Sparkline } from "./renderer.js";

const elements = {
  canvas: document.querySelector("#fieldCanvas"),
  spark: document.querySelector("#sparkCanvas"),
  playPause: document.querySelector("#playPause"),
  reset: document.querySelector("#reset"),
  experimentSelect: document.querySelector("#experimentSelect"),
  experimentTitle: document.querySelector("#experimentTitle"),
  experimentEyebrow: document.querySelector("#experimentEyebrow"),
  modelNote: document.querySelector("#modelNote"),
  gridSelect: document.querySelector("#gridSelect"),
  viewSelect: document.querySelector("#viewSelect"),
  modeButtons: [...document.querySelectorAll(".mode-button")],
  amplitude: document.querySelector("#amplitude"),
  frequency: document.querySelector("#frequency"),
  coupling: document.querySelector("#coupling"),
  diffusion: document.querySelector("#diffusion"),
  thermal: document.querySelector("#thermal"),
  variation: document.querySelector("#variation"),
  speed: document.querySelector("#speed"),
  metricMotors: document.querySelector("#metricMotors"),
  metricFps: document.querySelector("#metricFps"),
  metricTemp: document.querySelector("#metricTemp"),
  metricCurrent: document.querySelector("#metricCurrent"),
  metricEnergy: document.querySelector("#metricEnergy"),
  metricSaturated: document.querySelector("#metricSaturated"),
  metricAuxLabel: document.querySelector("#metricAuxLabel")
};

const initialGrid = parseGrid(elements.gridSelect.value);
const sheetSim = new MotorMuscleSim(initialGrid);
const humanoidSim = new HumanoidMotorSim({ motorTarget: initialGrid.cols * initialGrid.rows });
const fieldRenderer = new FieldRenderer(elements.canvas);
const humanoidRenderer = new HumanoidRenderer(elements.canvas);
const spark = new Sparkline(elements.spark);

let activeExperiment = elements.experimentSelect.value;
let sim = activeExperiment === "humanoid" ? humanoidSim : sheetSim;
let renderer = activeExperiment === "humanoid" ? humanoidRenderer : fieldRenderer;

let running = true;
let last = performance.now();
let frameCounter = 0;
let fps = 0;
let fpsTimer = performance.now();

function syncConfig() {
  sim.setConfig({
    amplitude: Number(elements.amplitude.value),
    frequency: Number(elements.frequency.value),
    coupling: Number(elements.coupling.value),
    diffusion: Number(elements.diffusion.value),
    thermalLoad: Number(elements.thermal.value),
    speed: Number(elements.speed.value)
  });
}

function setMode(mode) {
  sim.setConfig({ mode });
  for (const button of elements.modeButtons) {
    button.classList.toggle("active", button.dataset.mode === mode);
  }
}

function setGrid() {
  const grid = parseGrid(elements.gridSelect.value);
  if (activeExperiment === "humanoid") {
    sim.setConfig({
      motorTarget: grid.cols * grid.rows,
      variation: Number(elements.variation.value)
    });
  } else {
    sim.setConfig({
      ...grid,
      variation: Number(elements.variation.value)
    });
  }
}

function setExperiment(experiment) {
  activeExperiment = experiment;
  sim = activeExperiment === "humanoid" ? humanoidSim : sheetSim;
  renderer = activeExperiment === "humanoid" ? humanoidRenderer : fieldRenderer;
  elements.experimentTitle.textContent =
    activeExperiment === "humanoid" ? "微电机人形机器人仿真器" : "微电机复合肌体仿真器";
  elements.experimentEyebrow.textContent =
    activeExperiment === "humanoid" ? "mm-scale motor swarm humanoid" : "mm-scale motor swarm actuation";
  elements.modelNote.textContent =
    activeExperiment === "humanoid"
      ? "人形模型包含 206 块骨头，每块骨头附着执行器；顶层运动模式经自适应控制转换为局部微电机驱动。"
      : "平面模型把微电机视作可编程力/扭矩像素，使用低维协同模式、局部神经场扩散、热保护和弹性板耦合来模拟复合肌体。";
  syncConfig();
  setGrid();
}

function updateMetrics(stats) {
  elements.metricMotors.textContent = stats.motors.toLocaleString("en-US");
  elements.metricFps.textContent = Math.round(fps).toString();
  elements.metricTemp.textContent = stats.avgTemp.toFixed(1);
  elements.metricCurrent.textContent = stats.avgCurrent.toFixed(2);
  elements.metricEnergy.textContent = stats.energy.toFixed(1);
  if (activeExperiment === "humanoid") {
    elements.metricAuxLabel.textContent = "bones covered";
    elements.metricSaturated.textContent = `${stats.coveredBones}/${stats.bones}`;
  } else {
    elements.metricAuxLabel.textContent = "protected %";
    elements.metricSaturated.textContent = (stats.protectedRatio * 100).toFixed(1);
  }
}

function loop(now) {
  const delta = Math.min(0.05, (now - last) / 1000 || 1 / 60);
  last = now;
  if (running) {
    syncConfig();
    sim.step(delta);
  }
  renderer.draw(sim, elements.viewSelect.value);
  const stats = sim.stats();
  spark.push(stats);
  spark.draw();
  updateMetrics(stats);

  frameCounter += 1;
  if (now - fpsTimer > 500) {
    fps = (frameCounter * 1000) / (now - fpsTimer);
    frameCounter = 0;
    fpsTimer = now;
  }
  requestAnimationFrame(loop);
}

elements.playPause.addEventListener("click", () => {
  running = !running;
  elements.playPause.textContent = running ? "Ⅱ" : "▶";
  elements.playPause.setAttribute("aria-label", running ? "pause simulation" : "run simulation");
});

elements.reset.addEventListener("click", () => sim.resetMaterial());
elements.experimentSelect.addEventListener("change", () => setExperiment(elements.experimentSelect.value));
elements.gridSelect.addEventListener("change", setGrid);
elements.variation.addEventListener("input", setGrid);

for (const input of [
  elements.amplitude,
  elements.frequency,
  elements.coupling,
  elements.diffusion,
  elements.thermal,
  elements.speed
]) {
  input.addEventListener("input", syncConfig);
}

for (const button of elements.modeButtons) {
  button.addEventListener("click", () => setMode(button.dataset.mode));
}

syncConfig();
setMode("wave");
setExperiment(activeExperiment);
requestAnimationFrame(loop);
