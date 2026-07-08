function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function colorRamp(value, view) {
  const v = clamp(value, -1, 1);
  if (view === "temperature") {
    const hot = clamp((v + 1) * 0.5, 0, 1);
    const r = Math.round(34 + hot * 221);
    const g = Math.round(70 + Math.sin(hot * Math.PI) * 126);
    const b = Math.round(92 - hot * 64);
    return `rgb(${r},${g},${b})`;
  }
  if (view === "current") {
    const m = Math.abs(v);
    const r = Math.round(20 + m * 230);
    const g = Math.round(92 + m * 118);
    const b = Math.round(126 + (1 - m) * 88);
    return `rgb(${r},${g},${b})`;
  }
  if (view === "activation") {
    const pos = clamp(v, 0, 1);
    const neg = clamp(-v, 0, 1);
    return `rgb(${Math.round(34 + pos * 210)},${Math.round(76 + pos * 150 + neg * 35)},${Math.round(92 + neg * 140)})`;
  }
  if (view === "strain") {
    const s = clamp((v + 1) * 0.5, 0, 1);
    return `rgb(${Math.round(15 + s * 215)},${Math.round(94 + s * 110)},${Math.round(108 - s * 60)})`;
  }
  const positive = clamp(v, 0, 1);
  const negative = clamp(-v, 0, 1);
  const r = Math.round(18 + positive * 225 + negative * 20);
  const g = Math.round(64 + positive * 132 + negative * 68);
  const b = Math.round(84 + negative * 160);
  return `rgb(${r},${g},${b})`;
}

export class FieldRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const width = Math.max(320, Math.floor(rect.width * dpr));
    const height = Math.max(240, Math.floor(rect.height * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  draw(sim, view = "height") {
    this.resize();
    const ctx = this.ctx;
    const { width, height } = this.canvas;
    const cols = sim.cols;
    const rows = sim.rows;
    const field = sim.field(view);
    ctx.fillStyle = "#071114";
    ctx.fillRect(0, 0, width, height);

    const margin = Math.max(18, Math.min(width, height) * 0.035);
    const usableW = width - margin * 2;
    const usableH = height - margin * 2;
    const cellW = usableW / cols;
    const cellH = usableH / rows;

    let scale = 1;
    let offset = 0;
    if (view === "temperature") {
      offset = -68;
      scale = 1 / 34;
    } else if (view === "current") {
      scale = 1 / 5.5;
    } else if (view === "strain") {
      offset = -0.12;
      scale = 1 / 0.28;
    } else {
      scale = 1 / 1.8;
    }

    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        const i = y * cols + x;
        const value = (field[i] + offset) * scale;
        ctx.fillStyle = colorRamp(value, view);
        ctx.fillRect(
          margin + x * cellW,
          margin + y * cellH,
          Math.max(1, cellW + 0.5),
          Math.max(1, cellH + 0.5)
        );
      }
    }

    this.drawMeshOverlay(ctx, sim, margin, cellW, cellH);
  }

  drawMeshOverlay(ctx, sim, margin, cellW, cellH) {
    const step = Math.max(4, Math.round(sim.cols / 28));
    ctx.save();
    ctx.strokeStyle = "rgba(234, 242, 222, 0.18)";
    ctx.lineWidth = Math.max(1, Math.min(cellW, cellH) * 0.18);
    for (let x = 0; x < sim.cols; x += step) {
      ctx.beginPath();
      for (let y = 0; y < sim.rows; y += 1) {
        const i = y * sim.cols + x;
        const px = margin + (x + 0.5) * cellW;
        const py = margin + (y + 0.5) * cellH - sim.height[i] * Math.min(cellW, cellH) * 1.4;
        if (y === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    for (let y = 0; y < sim.rows; y += step) {
      ctx.beginPath();
      for (let x = 0; x < sim.cols; x += 1) {
        const i = y * sim.cols + x;
        const px = margin + (x + 0.5) * cellW;
        const py = margin + (y + 0.5) * cellH - sim.height[i] * Math.min(cellW, cellH) * 1.4;
        if (x === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    ctx.restore();
  }
}

export class HumanoidRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const width = Math.max(320, Math.floor(rect.width * dpr));
    const height = Math.max(240, Math.floor(rect.height * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  draw(sim, view = "height") {
    this.resize();
    const ctx = this.ctx;
    const { width, height } = this.canvas;
    ctx.fillStyle = "#071114";
    ctx.fillRect(0, 0, width, height);

    const pose = sim.pose();
    const scale = Math.min(width * 0.42, height * 0.58);
    const origin = { x: width * 0.5, y: height * 0.88 };
    const toScreen = (point) => ({
      x: origin.x + point.x * scale,
      y: origin.y - point.y * scale
    });

    this.drawGround(ctx, width, height, origin);
    this.drawMotorCloud(ctx, sim, pose, view, toScreen, scale);
    this.drawSkeleton(ctx, pose, toScreen);
    this.drawJointHud(ctx, sim, width, height);
  }

  drawGround(ctx, width, height, origin) {
    ctx.save();
    ctx.strokeStyle = "rgba(219, 211, 111, 0.18)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width * 0.12, origin.y);
    ctx.lineTo(width * 0.88, origin.y);
    ctx.stroke();
    for (let i = 0; i < 12; i += 1) {
      const x = width * (0.12 + i * 0.065);
      ctx.beginPath();
      ctx.moveTo(x, origin.y);
      ctx.lineTo(x + 18, origin.y + 16);
      ctx.stroke();
    }
    ctx.restore();
  }

  drawMotorCloud(ctx, sim, pose, view, toScreen, scale) {
    const field = sim.field(view);
    const point = Math.max(1.2, Math.min(4.5, scale * 0.007));
    for (const segment of sim.segments) {
      const geometry = pose.segments[segment.name];
      if (!geometry) continue;
      const along = { x: Math.cos(geometry.angle), y: Math.sin(geometry.angle) };
      const perp = { x: -along.y, y: along.x };
      for (let i = segment.start; i < segment.end; i += 1) {
        const s = (sim.localY[i] + 1) * 0.5;
        const lateral = sim.localX[i] * geometry.width * 0.5;
        const model = {
          x: geometry.start.x + along.x * geometry.length * s + perp.x * lateral,
          y: geometry.start.y + along.y * geometry.length * s + perp.y * lateral
        };
        const screen = toScreen(model);
        const value = this.normalizedValue(sim, field[i], view);
        ctx.fillStyle = colorRamp(value, view);
        ctx.fillRect(screen.x - point * 0.5, screen.y - point * 0.5, point, point);
      }
    }
  }

  normalizedValue(sim, value, view) {
    if (view === "temperature") return (value - 68) / 34;
    if (view === "current") return value / 5.5;
    if (view === "strain") return (value - 0.2) / 0.4;
    if (view === "activation") return value / 1.2;
    return value / 1.6;
  }

  drawSkeleton(ctx, pose, toScreen) {
    const n = pose.nodes;
    const bones = [
      ["leftHip", "root"],
      ["root", "rightHip"],
      ["root", "torsoTop"],
      ["torsoTop", "leftShoulder"],
      ["torsoTop", "rightShoulder"],
      ["leftShoulder", "lElbow"],
      ["lElbow", "lWrist"],
      ["rightShoulder", "rElbow"],
      ["rElbow", "rWrist"],
      ["leftHip", "lKnee"],
      ["lKnee", "lAnkle"],
      ["rightHip", "rKnee"],
      ["rKnee", "rAnkle"]
    ];
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(237, 242, 228, 0.55)";
    ctx.lineWidth = 2;
    for (const [a, b] of bones) {
      const pa = toScreen(n[a]);
      const pb = toScreen(n[b]);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(219, 211, 111, 0.95)";
    for (const key of Object.keys(n)) {
      const p = toScreen(n[key]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  drawJointHud(ctx, sim, width, height) {
    const left = width * 0.035;
    const top = height * 0.045;
    const barWidth = Math.min(220, width * 0.22);
    ctx.save();
    ctx.font = "12px Avenir Next, sans-serif";
    ctx.fillStyle = "rgba(237, 242, 228, 0.72)";
    const covered = sim.boneMotorCounts ? [...sim.boneMotorCounts].filter((count) => count > 0).length : 0;
    const adaptiveMean = sim.adaptiveGain
      ? [...sim.adaptiveGain].reduce((sum, value) => sum + value, 0) / sim.adaptiveGain.length
      : 1;
    ctx.fillText(`206-bone actuator map: ${covered}/${sim.boneCount || 0}`, left, top);
    ctx.fillText(`adaptive top control gain: ${adaptiveMean.toFixed(2)}`, left, top + 16);
    for (let i = 0; i < Math.min(8, sim.joints.length); i += 1) {
      const joint = sim.joints[i];
      const y = top + 42 + i * 15;
      const value = clamp(joint.angle / Math.max(Math.abs(joint.min), Math.abs(joint.max)), -1, 1);
      ctx.fillStyle = "rgba(145, 164, 157, 0.55)";
      ctx.fillRect(left, y, barWidth, 2);
      ctx.fillStyle = value >= 0 ? "#e96f43" : "#6da7d8";
      const w = Math.abs(value) * barWidth * 0.5;
      const center = left + barWidth * 0.5;
      ctx.fillRect(value >= 0 ? center : center - w, y - 2, w, 6);
      ctx.fillStyle = "rgba(237, 242, 228, 0.68)";
      ctx.fillText(joint.name, left + barWidth + 10, y + 4);
    }
    ctx.restore();
  }
}

export class Sparkline {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.history = [];
  }

  push(stats) {
    this.history.push({
      temp: stats.avgTemp,
      current: stats.avgCurrent,
      strain: stats.meanStrain * 80
    });
    if (this.history.length > 160) this.history.shift();
  }

  draw() {
    const ctx = this.ctx;
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(240, Math.floor(rect.width * dpr));
    const height = Math.max(90, Math.floor(rect.height * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#101b1d";
    ctx.fillRect(0, 0, width, height);
    this.line("temp", 24, 110, "#e96f43", width, height);
    this.line("current", 0, 7, "#dbd36f", width, height);
    this.line("strain", 0, 16, "#6da7d8", width, height);
  }

  line(key, min, max, color, width, height) {
    if (this.history.length < 2) return;
    const ctx = this.ctx;
    ctx.beginPath();
    for (let i = 0; i < this.history.length; i += 1) {
      const x = (i / (this.history.length - 1)) * width;
      const y = height - clamp((this.history[i][key] - min) / (max - min), 0, 1) * height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}
