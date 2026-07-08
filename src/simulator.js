const TWO_PI = Math.PI * 2;

export const DEFAULT_CONFIG = Object.freeze({
  cols: 96,
  rows: 64,
  spacingMm: 1.0,
  dt: 1 / 240,
  mode: "wave",
  amplitude: 1.0,
  frequency: 0.8,
  coupling: 42,
  diffusion: 6,
  thermalLoad: 1.0,
  variation: 0.08,
  speed: 1.0
});

function hashNoise(index, seed = 13) {
  let x = (index + 1) * 1103515245 + seed * 12345;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return ((x >>> 0) / 4294967295) * 2 - 1;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export class MotorMuscleSim {
  constructor(config = {}) {
    this.configure({ ...DEFAULT_CONFIG, ...config });
  }

  configure(config) {
    this.config = { ...DEFAULT_CONFIG, ...this.config, ...config };
    this.cols = this.config.cols;
    this.rows = this.config.rows;
    this.count = this.cols * this.rows;
    this.time = 0;
    this.energy = 0;

    const n = this.count;
    this.height = new Float32Array(n);
    this.velocity = new Float32Array(n);
    this.activation = new Float32Array(n);
    this.nextActivation = new Float32Array(n);
    this.current = new Float32Array(n);
    this.temperature = new Float32Array(n);
    this.strain = new Float32Array(n);
    this.voltage = new Float32Array(n);
    this.torqueGain = new Float32Array(n);
    this.resistance = new Float32Array(n);
    this.mass = new Float32Array(n);
    this.protection = new Float32Array(n);

    this.resetMaterial();
  }

  resetMaterial() {
    const variation = this.config.variation;
    for (let i = 0; i < this.count; i += 1) {
      const n1 = hashNoise(i, 5);
      const n2 = hashNoise(i, 17);
      const n3 = hashNoise(i, 41);
      this.height[i] = 0;
      this.velocity[i] = 0;
      this.activation[i] = 0;
      this.nextActivation[i] = 0;
      this.current[i] = 0;
      this.temperature[i] = 24;
      this.strain[i] = 0;
      this.voltage[i] = 0;
      this.torqueGain[i] = 1 + n1 * variation;
      this.resistance[i] = 1.8 * (1 + n2 * variation);
      this.mass[i] = 0.018 * (1 + n3 * variation * 0.5);
      this.protection[i] = 0;
    }
    this.time = 0;
    this.energy = 0;
  }

  setConfig(partial) {
    const needsRebuild =
      partial.cols !== undefined ||
      partial.rows !== undefined ||
      partial.variation !== undefined;
    this.config = { ...this.config, ...partial };
    if (needsRebuild) {
      this.configure(this.config);
    }
  }

  step(frameDt = this.config.dt) {
    const substeps = Math.max(1, Math.round(this.config.speed * 2));
    const dt = Math.min(1 / 60, frameDt * this.config.speed) / substeps;
    for (let s = 0; s < substeps; s += 1) {
      this.integrate(dt);
    }
  }

  integrate(dt) {
    const {
      amplitude,
      frequency,
      coupling,
      diffusion,
      thermalLoad,
      mode
    } = this.config;
    const cols = this.cols;
    const rows = this.rows;
    const invCols = 1 / Math.max(1, cols - 1);
    const invRows = 1 / Math.max(1, rows - 1);
    const t = this.time;
    const n = this.count;

    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        const i = y * cols + x;
        const left = x > 0 ? i - 1 : i;
        const right = x < cols - 1 ? i + 1 : i;
        const up = y > 0 ? i - cols : i;
        const down = y < rows - 1 ? i + cols : i;

        const lapActivation =
          this.activation[left] + this.activation[right] + this.activation[up] + this.activation[down] - 4 * this.activation[i];
        const lapHeight =
          this.height[left] + this.height[right] + this.height[up] + this.height[down] - 4 * this.height[i];

        const nx = x * invCols * 2 - 1;
        const ny = y * invRows * 2 - 1;
        const desired = this.pattern(mode, nx, ny, t, frequency) * amplitude;
        const feedback = -0.34 * this.height[i] - 0.06 * this.velocity[i];
        const heatInhibition = clamp((this.temperature[i] - 72) / 28, 0, 0.9);
        const da = (-this.activation[i] + desired + diffusion * lapActivation * 0.025 + feedback) / 0.055;
        this.nextActivation[i] = clamp(this.activation[i] + da * dt, -1.4, 1.4) * (1 - heatInhibition);

        const localStrain = Math.sqrt(
          (this.height[right] - this.height[left]) ** 2 +
          (this.height[down] - this.height[up]) ** 2
        ) * 0.5;
        this.strain[i] = localStrain;

        const protection = clamp((this.temperature[i] - 80) / 35 + localStrain * 0.18, 0, 1);
        this.protection[i] = protection;
        const targetVoltage = 8.0 * this.nextActivation[i] * (1 - protection);
        const backEmf = 0.035 * this.velocity[i];
        const rT = this.resistance[i] * (1 + 0.0039 * (this.temperature[i] - 24));
        const di = (targetVoltage - rT * this.current[i] - backEmf) / 0.012;
        this.current[i] += di * dt;
        this.voltage[i] = targetVoltage;

        const edgePin =
          x === 0 || y === 0 || x === cols - 1 || y === rows - 1 ? 22 : 0;
        const actuatorForce = 2.65 * this.torqueGain[i] * this.current[i];
        const sheetForce =
          coupling * lapHeight - (18 + edgePin) * this.height[i] - 0.92 * this.velocity[i];
        const nonlinearMembrane = -6.5 * this.height[i] ** 3 - 1.25 * localStrain * this.height[i];
        const acceleration = (actuatorForce + sheetForce + nonlinearMembrane) / this.mass[i];
        this.velocity[i] = clamp(this.velocity[i] + acceleration * dt, -44, 44);
        this.height[i] = clamp(this.height[i] + this.velocity[i] * dt, -2.6, 2.6);

        const joule = this.current[i] * this.current[i] * rT * thermalLoad;
        const mechLoss = 0.018 * Math.abs(this.velocity[i] * actuatorForce);
        const cooling = 0.16 * (this.temperature[i] - 24);
        this.temperature[i] += (joule + mechLoss - cooling) * dt / 0.95;
        this.temperature[i] = Math.max(20, this.temperature[i]);
        this.energy += Math.abs(targetVoltage * this.current[i]) * dt / n;
      }
    }

    const temp = this.activation;
    this.activation = this.nextActivation;
    this.nextActivation = temp;
    this.time += dt;
  }

  pattern(mode, nx, ny, time, frequency) {
    const phase = TWO_PI * frequency * time;
    const r = Math.sqrt(nx * nx + ny * ny);
    const theta = Math.atan2(ny, nx);

    switch (mode) {
      case "curl":
        return Math.tanh(1.7 * (nx + 0.38 * Math.sin(phase))) + 0.25 * Math.sin(phase + ny * Math.PI);
      case "grasp":
        return Math.cos(Math.PI * r * 1.35 - phase) * Math.exp(-r * 0.72) - 0.26 * r;
      case "twist":
        return Math.sin(theta * 2 + phase) * (0.35 + 0.9 * r);
      case "peristaltic":
        return Math.sin((nx * 2.8 + Math.sin(ny * Math.PI)) * Math.PI - phase) * (0.7 + 0.3 * Math.cos(ny * Math.PI));
      case "focus": {
        const cx = 0.45 * Math.sin(phase * 0.41);
        const cy = 0.45 * Math.cos(phase * 0.37);
        const d = (nx - cx) ** 2 + (ny - cy) ** 2;
        return 1.35 * Math.exp(-d * 8.5) - 0.42 * Math.exp(-r * 1.2);
      }
      case "wave":
      default:
        return Math.sin((nx * 1.7 + ny * 0.45) * Math.PI * 2 - phase) * (0.84 + 0.16 * Math.cos(ny * Math.PI));
    }
  }

  stats() {
    let temp = 0;
    let current = 0;
    let protectedCount = 0;
    let maxAbsHeight = 0;
    let meanStrain = 0;
    for (let i = 0; i < this.count; i += 1) {
      temp += this.temperature[i];
      current += Math.abs(this.current[i]);
      meanStrain += this.strain[i];
      maxAbsHeight = Math.max(maxAbsHeight, Math.abs(this.height[i]));
      if (this.protection[i] > 0.05) protectedCount += 1;
    }
    return {
      motors: this.count,
      time: this.time,
      avgTemp: temp / this.count,
      avgCurrent: current / this.count,
      energy: this.energy,
      protectedRatio: protectedCount / this.count,
      maxAbsHeight,
      meanStrain: meanStrain / this.count
    };
  }

  field(view) {
    switch (view) {
      case "activation":
        return this.activation;
      case "temperature":
        return this.temperature;
      case "current":
        return this.current;
      case "strain":
        return this.strain;
      case "height":
      default:
        return this.height;
    }
  }
}

export function parseGrid(value) {
  const [cols, rows] = String(value).split("x").map(Number);
  if (!Number.isFinite(cols) || !Number.isFinite(rows) || cols <= 0 || rows <= 0) {
    throw new Error(`Invalid grid: ${value}`);
  }
  return { cols, rows };
}
