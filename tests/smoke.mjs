import { strict as assert } from "node:assert";
import { ANATOMY_206, anatomySummary } from "../src/anatomy206.js";
import { HumanoidMotorSim } from "../src/humanoid.js";
import { MotorMuscleSim } from "../src/simulator.js";

function assertFiniteArray(name, array) {
  for (let i = 0; i < array.length; i += Math.max(1, Math.floor(array.length / 257))) {
    assert(Number.isFinite(array[i]), `${name}[${i}] is not finite`);
  }
}

const sim = new MotorMuscleSim({
  cols: 96,
  rows: 64,
  mode: "peristaltic",
  amplitude: 1.2,
  frequency: 1.1,
  coupling: 48,
  diffusion: 7,
  thermalLoad: 1.2,
  variation: 0.1,
  speed: 1
});

for (let i = 0; i < 240; i += 1) {
  sim.step(1 / 120);
}

const stats = sim.stats();
assert.equal(stats.motors, 6144);
assert(stats.avgTemp > 23, "temperature should evolve above ambient");
assert(stats.avgCurrent > 0.01, "motors should draw current");
assert(stats.maxAbsHeight > 0.01, "body should deform");
assert(stats.energy > 0, "energy should accumulate");
assertFiniteArray("height", sim.height);
assertFiniteArray("activation", sim.activation);
assertFiniteArray("temperature", sim.temperature);
assertFiniteArray("current", sim.current);

const larger = new MotorMuscleSim({ cols: 160, rows: 120, mode: "focus" });
larger.step(1 / 60);
assert.equal(larger.stats().motors, 19200);
assertFiniteArray("large.height", larger.height);

const humanoid = new HumanoidMotorSim({
  motorTarget: 6144,
  mode: "wave",
  amplitude: 1.15,
  frequency: 0.9,
  coupling: 44,
  diffusion: 7,
  thermalLoad: 1.1,
  variation: 0.08,
  speed: 1
});

for (let i = 0; i < 300; i += 1) {
  humanoid.step(1 / 120);
}

const humanoidStats = humanoid.stats();
const summary = anatomySummary();
assert.equal(ANATOMY_206.length, 206, "anatomy catalog should model 206 bones");
assert.equal(summary.totalBones, 206, "anatomy summary should report 206 bones");
assert(humanoidStats.motors >= 5800, "humanoid should contain thousands of motors");
assert.equal(humanoidStats.bones, 206, "humanoid should expose 206 bones");
assert.equal(humanoidStats.coveredBones, 206, "every bone should have attached motor actuators");
assert(humanoidStats.avgTemp > 23, "humanoid motors should heat above ambient");
assert(humanoidStats.avgCurrent > 0.01, "humanoid motors should draw current");
assert(humanoidStats.energy > 0, "humanoid energy should accumulate");
assert(humanoidStats.activeJoints >= 4, "humanoid should move several joints");
assert(Math.abs(humanoidStats.adaptiveGainMean - 1) > 0.001, "adaptive controller gains should update during motion");
assert(humanoid.pose().segments.torso, "humanoid should expose torso pose");
assert(humanoid.boneMotorCounts.every((count) => count > 0), "all 206 bones should have motors");
assertFiniteArray("humanoid.boneIndex", humanoid.boneIndex);
assertFiniteArray("humanoid.activation", humanoid.activation);
assertFiniteArray("humanoid.temperature", humanoid.temperature);
assertFiniteArray("humanoid.current", humanoid.current);
assertFiniteArray("humanoid.deformation", humanoid.deformation);

const denseHumanoid = new HumanoidMotorSim({ motorTarget: 19200, mode: "grasp" });
denseHumanoid.step(1 / 60);
assert(denseHumanoid.stats().motors >= 17000, "dense humanoid should stay in high-density range");
assertFiniteArray("denseHumanoid.current", denseHumanoid.current);

console.log(JSON.stringify({
  ok: true,
  motors: stats.motors,
  avgTemp: Number(stats.avgTemp.toFixed(3)),
  avgCurrent: Number(stats.avgCurrent.toFixed(3)),
  maxAbsHeight: Number(stats.maxAbsHeight.toFixed(3)),
  energy: Number(stats.energy.toFixed(3)),
  largeMotors: larger.stats().motors,
  humanoidMotors: humanoidStats.motors,
  humanoidBones: humanoidStats.bones,
  humanoidCoveredBones: humanoidStats.coveredBones,
  humanoidActiveJoints: humanoidStats.activeJoints,
  humanoidAdaptiveGainMean: Number(humanoidStats.adaptiveGainMean.toFixed(4)),
  humanoidAvgTemp: Number(humanoidStats.avgTemp.toFixed(3)),
  denseHumanoidMotors: denseHumanoid.stats().motors
}, null, 2));
