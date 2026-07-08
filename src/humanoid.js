import { ANATOMY_206, anatomySummary } from "./anatomy206.js";

const TWO_PI = Math.PI * 2;

const BASE_SEGMENTS = [
  { name: "head", label: "head", cols: 18, rows: 18, length: 0.20, width: 0.16, joint: "neck" },
  { name: "torso", label: "torso", cols: 34, rows: 44, length: 0.54, width: 0.30, joint: "spine" },
  { name: "pelvis", label: "pelvis", cols: 30, rows: 16, length: 0.32, width: 0.16, joint: "spine" },
  { name: "lUpperArm", label: "left upper arm", cols: 14, rows: 30, length: 0.32, width: 0.09, joint: "lShoulder" },
  { name: "rUpperArm", label: "right upper arm", cols: 14, rows: 30, length: 0.32, width: 0.09, joint: "rShoulder" },
  { name: "lForearm", label: "left forearm", cols: 12, rows: 28, length: 0.30, width: 0.075, joint: "lElbow" },
  { name: "rForearm", label: "right forearm", cols: 12, rows: 28, length: 0.30, width: 0.075, joint: "rElbow" },
  { name: "lHand", label: "left hand", cols: 10, rows: 12, length: 0.10, width: 0.08, joint: "lElbow" },
  { name: "rHand", label: "right hand", cols: 10, rows: 12, length: 0.10, width: 0.08, joint: "rElbow" },
  { name: "lThigh", label: "left thigh", cols: 16, rows: 36, length: 0.45, width: 0.12, joint: "lHip" },
  { name: "rThigh", label: "right thigh", cols: 16, rows: 36, length: 0.45, width: 0.12, joint: "rHip" },
  { name: "lShin", label: "left shin", cols: 14, rows: 36, length: 0.43, width: 0.095, joint: "lKnee" },
  { name: "rShin", label: "right shin", cols: 14, rows: 36, length: 0.43, width: 0.095, joint: "rKnee" },
  { name: "lFoot", label: "left foot", cols: 18, rows: 10, length: 0.20, width: 0.08, joint: "lAnkle" },
  { name: "rFoot", label: "right foot", cols: 18, rows: 10, length: 0.20, width: 0.08, joint: "rAnkle" }
];

const JOINTS = [
  { name: "spine", inertia: 1.8, damping: 2.6, stiffness: 3.2, min: -0.55, max: 0.55 },
  { name: "neck", inertia: 0.5, damping: 1.8, stiffness: 2.4, min: -0.42, max: 0.42 },
  { name: "lShoulder", inertia: 0.9, damping: 1.7, stiffness: 1.2, min: -1.2, max: 1.4 },
  { name: "rShoulder", inertia: 0.9, damping: 1.7, stiffness: 1.2, min: -1.2, max: 1.4 },
  { name: "lElbow", inertia: 0.55, damping: 1.35, stiffness: 0.9, min: -0.25, max: 1.8 },
  { name: "rElbow", inertia: 0.55, damping: 1.35, stiffness: 0.9, min: -0.25, max: 1.8 },
  { name: "lHip", inertia: 1.25, damping: 2.0, stiffness: 1.6, min: -0.95, max: 1.05 },
  { name: "rHip", inertia: 1.25, damping: 2.0, stiffness: 1.6, min: -0.95, max: 1.05 },
  { name: "lKnee", inertia: 0.85, damping: 1.8, stiffness: 1.25, min: -0.12, max: 1.65 },
  { name: "rKnee", inertia: 0.85, damping: 1.8, stiffness: 1.25, min: -0.12, max: 1.65 },
  { name: "lAnkle", inertia: 0.45, damping: 1.2, stiffness: 1.9, min: -0.65, max: 0.65 },
  { name: "rAnkle", inertia: 0.45, damping: 1.2, stiffness: 1.9, min: -0.65, max: 0.65 }
];

export const HUMANOID_DEFAULT_CONFIG = Object.freeze({
  motorTarget: 6144,
  dt: 1 / 240,
  mode: "wave",
  amplitude: 1.0,
  frequency: 0.82,
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

function vector(angle, length) {
  return { x: Math.cos(angle) * length, y: Math.sin(angle) * length };
}

function add(a, b) {
  return { x: a.x + b.x, y: a.y + b.y };
}

function clampAngle(joint, value) {
  return clamp(value, joint.min, joint.max);
}

export class HumanoidMotorSim {
  constructor(config = {}) {
    this.configure({ ...HUMANOID_DEFAULT_CONFIG, ...config });
  }

  configure(config) {
    this.config = { ...HUMANOID_DEFAULT_CONFIG, ...this.config, ...config };
    this.time = 0;
    this.energy = 0;
    this.buildSegments();
    this.buildBones();
    this.buildMotors();
    this.buildJoints();
    this.resetMaterial();
  }

  buildSegments() {
    const baseCount = BASE_SEGMENTS.reduce((sum, segment) => sum + segment.cols * segment.rows, 0);
    const factor = Math.sqrt(Math.max(0.25, this.config.motorTarget / baseCount));
    let start = 0;
    this.segments = BASE_SEGMENTS.map((segment, index) => {
      const cols = Math.max(6, Math.round(segment.cols * factor));
      const rows = Math.max(6, Math.round(segment.rows * factor));
      const count = cols * rows;
      const built = {
        ...segment,
        index,
        cols,
        rows,
        start,
        count,
        end: start + count
      };
      start += count;
      return built;
    });
    this.count = start;
  }

  buildBones() {
    this.bones = ANATOMY_206.map((bone) => ({ ...bone }));
    this.boneCount = this.bones.length;
    this.anatomySummary = anatomySummary(this.bones);
    this.bonesBySegment = new Map();
    for (const bone of this.bones) {
      if (!this.bonesBySegment.has(bone.segment)) this.bonesBySegment.set(bone.segment, []);
      this.bonesBySegment.get(bone.segment).push(bone.id);
    }
  }

  buildMotors() {
    const n = this.count;
    this.localX = new Float32Array(n);
    this.localY = new Float32Array(n);
    this.segmentIndex = new Int16Array(n);
    this.boneIndex = new Int16Array(n);
    this.jointIndex = new Int16Array(n);
    this.motorSign = new Float32Array(n);
    this.momentArm = new Float32Array(n);
    this.activation = new Float32Array(n);
    this.nextActivation = new Float32Array(n);
    this.current = new Float32Array(n);
    this.temperature = new Float32Array(n);
    this.protection = new Float32Array(n);
    this.strain = new Float32Array(n);
    this.deformation = new Float32Array(n);
    this.torqueGain = new Float32Array(n);
    this.resistance = new Float32Array(n);
    this.boneMotorCounts = new Int32Array(this.boneCount);

    const jointLookup = new Map(JOINTS.map((joint, index) => [joint.name, index]));
    for (const segment of this.segments) {
      const segmentBones = this.bonesBySegment.get(segment.name) || [0];
      for (let y = 0; y < segment.rows; y += 1) {
        for (let x = 0; x < segment.cols; x += 1) {
          const i = segment.start + y * segment.cols + x;
          const localIndex = y * segment.cols + x;
          const boneId = segmentBones[localIndex % segmentBones.length];
          const bone = this.bones[boneId];
          const nx = segment.cols === 1 ? 0 : (x / (segment.cols - 1)) * 2 - 1;
          const ny = segment.rows === 1 ? 0 : (y / (segment.rows - 1)) * 2 - 1;
          this.localX[i] = nx;
          this.localY[i] = ny;
          this.segmentIndex[i] = segment.index;
          this.boneIndex[i] = boneId;
          this.boneMotorCounts[boneId] += 1;
          this.jointIndex[i] = jointLookup.get(bone.joint) ?? jointLookup.get(segment.joint) ?? 0;
          this.motorSign[i] = nx < 0 ? -1 : 1;
          this.momentArm[i] = (0.006 + Math.abs(nx) * 0.012 + (1 - Math.abs(ny)) * 0.004) * (0.85 + bone.weight * 0.12);
        }
      }
    }
  }

  buildJoints() {
    this.joints = JOINTS.map((joint) => ({ ...joint, angle: 0, velocity: 0, target: 0, torque: 0 }));
    this.jointLookup = new Map(this.joints.map((joint, index) => [joint.name, index]));
    this.jointMotorCounts = new Float32Array(this.joints.length);
    this.adaptiveGain = new Float32Array(this.joints.length);
    this.adaptiveBias = new Float32Array(this.joints.length);
    this.jointStress = new Float32Array(this.joints.length);
    this.trackingError = new Float32Array(this.joints.length);
    for (let i = 0; i < this.count; i += 1) {
      this.jointMotorCounts[this.jointIndex[i]] += 1;
    }
  }

  resetMaterial() {
    const variation = this.config.variation;
    for (let i = 0; i < this.count; i += 1) {
      this.activation[i] = 0;
      this.nextActivation[i] = 0;
      this.current[i] = 0;
      this.temperature[i] = 24;
      this.protection[i] = 0;
      this.strain[i] = 0;
      this.deformation[i] = 0;
      this.torqueGain[i] = 1 + hashNoise(i, 19) * variation;
      this.resistance[i] = 2.1 * (1 + hashNoise(i, 29) * variation);
    }
    for (const joint of this.joints) {
      joint.angle = 0;
      joint.velocity = 0;
      joint.target = 0;
      joint.torque = 0;
      joint.trackingError = 0;
      joint.adaptiveGain = 1;
      joint.adaptiveStress = 0;
    }
    for (let j = 0; j < this.joints.length; j += 1) {
      this.adaptiveGain[j] = 1;
      this.adaptiveBias[j] = 0;
      this.jointStress[j] = 0;
      this.trackingError[j] = 0;
    }
    this.time = 0;
    this.energy = 0;
  }

  setConfig(partial) {
    const needsRebuild =
      partial.motorTarget !== undefined ||
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
    const t = this.time;
    const phase = TWO_PI * this.config.frequency * t;
    const jointIntents = this.computeJointIntents(phase, dt);
    for (const joint of this.joints) {
      joint.torque = 0;
    }

    for (let i = 0; i < this.count; i += 1) {
      const segment = this.segments[this.segmentIndex[i]];
      const jointId = this.jointIndex[i];
      const joint = this.joints[jointId];
      const bone = this.bones[this.boneIndex[i]];
      const localWave = Math.sin(phase + this.localY[i] * Math.PI * 1.7 + bone.id * 0.071 + segment.index * 0.37);
      const reflex = -0.18 * joint.angle - 0.025 * joint.velocity;
      const intent = jointIntents[jointId] + reflex;
      const heatInhibition = clamp((this.temperature[i] - 74) / 30, 0, 0.92);
      const boneDrive = 0.88 + bone.weight * 0.08;
      const desired = (intent * this.motorSign[i] * boneDrive + localWave * 0.16 * this.config.diffusion / 10) * this.config.amplitude;
      const da = (-this.activation[i] + desired) / 0.045;
      this.nextActivation[i] = clamp(this.activation[i] + da * dt, -1.35, 1.35) * (1 - heatInhibition);

      const localStrain = Math.abs(joint.angle) * 0.32 + Math.abs(joint.velocity) * 0.018 + Math.abs(this.localY[i]) * 0.025;
      const protection = clamp((this.temperature[i] - 82) / 34 + localStrain * 0.55, 0, 1);
      this.protection[i] = protection;
      this.strain[i] = localStrain;

      const targetVoltage = 4.8 * this.nextActivation[i] * (1 - protection);
      const rT = this.resistance[i] * (1 + 0.0039 * (this.temperature[i] - 24));
      const backEmf = 0.03 * joint.velocity * this.motorSign[i];
      const di = (targetVoltage - rT * this.current[i] - backEmf) / 0.011;
      this.current[i] = clamp(this.current[i] + di * dt, -4.5, 4.5);

      const torque = this.current[i] * this.motorSign[i] * this.momentArm[i] * this.torqueGain[i] * 5.6;
      joint.torque += torque;

      this.deformation[i] = clamp(
        this.activation[i] * 0.6 + joint.angle * this.motorSign[i] * 0.7 + localWave * 0.18,
        -1.8,
        1.8
      );

      const joule = this.current[i] * this.current[i] * rT * this.config.thermalLoad * 0.18;
      const mechanicalLoss = Math.abs(torque * joint.velocity) * 0.08;
      const cooling = 0.55 * (this.temperature[i] - 24);
      this.temperature[i] += (joule + mechanicalLoss - cooling) * dt / 1.4;
      this.temperature[i] = Math.max(20, this.temperature[i]);
      this.energy += Math.abs(targetVoltage * this.current[i]) * dt / this.count;
    }

    for (let j = 0; j < this.joints.length; j += 1) {
      const joint = this.joints[j];
      const normalizedTorque = joint.torque / Math.max(1, Math.sqrt(this.jointMotorCounts[j]));
      const passive = -joint.stiffness * joint.angle - joint.damping * joint.velocity;
      const acceleration = (normalizedTorque + passive) / joint.inertia;
      joint.velocity = clamp(joint.velocity + acceleration * dt, -5.5, 5.5);
      joint.angle = clampAngle(joint, joint.angle + joint.velocity * dt);
      if (joint.angle === joint.min || joint.angle === joint.max) {
        joint.velocity *= -0.12;
      }
    }

    const temp = this.activation;
    this.activation = this.nextActivation;
    this.nextActivation = temp;
    this.time += dt;
  }

  computeJointIntents(phase, dt) {
    const targets = new Float32Array(this.joints.length);
    const targetFor = (name, value) => {
      const index = this.jointLookup.get(name);
      const joint = this.joints[index];
      targets[index] = clampAngle(joint, value * this.config.amplitude);
    };

    const gait = this.config.mode === "peristaltic" ? 1.35 : 1;
    if (this.config.mode === "curl") {
      targetFor("spine", 0.18 + 0.08 * Math.sin(phase));
      targetFor("lHip", 0.42 + 0.08 * Math.sin(phase));
      targetFor("rHip", 0.42 + 0.08 * Math.sin(phase));
      targetFor("lKnee", 0.82 + 0.16 * Math.sin(phase + 0.4));
      targetFor("rKnee", 0.82 + 0.16 * Math.sin(phase + 0.4));
      targetFor("lAnkle", -0.22);
      targetFor("rAnkle", -0.22);
      targetFor("lShoulder", 0.18);
      targetFor("rShoulder", 0.18);
      targetFor("lElbow", 0.42);
      targetFor("rElbow", 0.42);
    } else if (this.config.mode === "grasp") {
      targetFor("spine", 0.06 * Math.sin(phase * 0.5));
      targetFor("neck", 0.08 * Math.sin(phase * 0.4));
      targetFor("lShoulder", 0.86 + 0.14 * Math.sin(phase));
      targetFor("rShoulder", 0.86 + 0.14 * Math.sin(phase));
      targetFor("lElbow", 0.78 + 0.18 * Math.sin(phase + 0.5));
      targetFor("rElbow", 0.78 + 0.18 * Math.sin(phase + 0.5));
      targetFor("lHip", -0.08);
      targetFor("rHip", -0.08);
    } else if (this.config.mode === "twist") {
      targetFor("spine", 0.36 * Math.sin(phase));
      targetFor("neck", -0.18 * Math.sin(phase));
      targetFor("lShoulder", 0.42 * Math.sin(phase + Math.PI));
      targetFor("rShoulder", 0.42 * Math.sin(phase));
      targetFor("lElbow", 0.52 + 0.18 * Math.sin(phase));
      targetFor("rElbow", 0.52 - 0.18 * Math.sin(phase));
      targetFor("lHip", -0.16 * Math.sin(phase));
      targetFor("rHip", 0.16 * Math.sin(phase));
    } else if (this.config.mode === "focus") {
      targetFor("spine", 0.05 * Math.sin(phase * 1.7));
      targetFor("neck", 0.06 * Math.sin(phase * 1.1));
      targetFor("lAnkle", 0.18 * Math.sin(phase));
      targetFor("rAnkle", -0.18 * Math.sin(phase));
      targetFor("lHip", 0.08 * Math.sin(phase + 0.4));
      targetFor("rHip", -0.08 * Math.sin(phase + 0.4));
      targetFor("lKnee", 0.18 + 0.05 * Math.sin(phase));
      targetFor("rKnee", 0.18 - 0.05 * Math.sin(phase));
    } else {
      targetFor("spine", 0.07 * Math.sin(phase * 0.5));
      targetFor("neck", 0.05 * Math.sin(phase * 0.6));
      targetFor("lHip", 0.36 * Math.sin(phase) * gait);
      targetFor("rHip", 0.36 * Math.sin(phase + Math.PI) * gait);
      targetFor("lKnee", Math.max(0, 0.55 * Math.sin(phase + Math.PI / 2)) * gait);
      targetFor("rKnee", Math.max(0, 0.55 * Math.sin(phase + Math.PI * 1.5)) * gait);
      targetFor("lAnkle", -0.18 * Math.sin(phase + Math.PI / 2));
      targetFor("rAnkle", -0.18 * Math.sin(phase + Math.PI * 1.5));
      targetFor("lShoulder", -0.32 * Math.sin(phase));
      targetFor("rShoulder", -0.32 * Math.sin(phase + Math.PI));
      targetFor("lElbow", 0.36 + 0.12 * Math.sin(phase + Math.PI));
      targetFor("rElbow", 0.36 + 0.12 * Math.sin(phase));
    }
    return this.adaptJointIntents(targets, dt);
  }

  adaptJointIntents(targets, dt) {
    const intents = new Float32Array(this.joints.length);
    this.jointStress.fill(0);
    const counts = new Float32Array(this.joints.length);
    for (let i = 0; i < this.count; i += 1) {
      const jointId = this.jointIndex[i];
      this.jointStress[jointId] += this.protection[i] + clamp((this.temperature[i] - 40) / 60, 0, 1);
      counts[jointId] += 1;
    }
    for (let j = 0; j < this.joints.length; j += 1) {
      const joint = this.joints[j];
      const stress = counts[j] > 0 ? this.jointStress[j] / counts[j] : 0;
      const error = targets[j] - joint.angle;
      this.trackingError[j] = error;

      const gain = this.adaptiveGain[j];
      const learning = Math.abs(error) * 0.72 - stress * 0.32 - (gain - 1) * 0.035;
      this.adaptiveGain[j] = clamp(gain + learning * dt, 0.62, 2.45);
      this.adaptiveBias[j] = clamp(this.adaptiveBias[j] + (error * 0.16 - this.adaptiveBias[j] * 0.08) * dt, -0.32, 0.32);

      joint.target = targets[j];
      joint.trackingError = error;
      joint.adaptiveGain = this.adaptiveGain[j];
      joint.adaptiveStress = stress;
      intents[j] = clamp(
        error * 2.05 * this.adaptiveGain[j] - joint.velocity * (0.12 + 0.05 * this.adaptiveGain[j]) + this.adaptiveBias[j],
        -1.25,
        1.25
      );
    }
    return intents;
  }

  stats() {
    let temp = 0;
    let current = 0;
    let protectedCount = 0;
    let meanStrain = 0;
    let jointActivity = 0;
    let adaptiveGain = 0;
    let trackingError = 0;
    let coveredBones = 0;
    for (let i = 0; i < this.count; i += 1) {
      temp += this.temperature[i];
      current += Math.abs(this.current[i]);
      meanStrain += this.strain[i];
      if (this.protection[i] > 0.05) protectedCount += 1;
    }
    for (const joint of this.joints) {
      jointActivity += Math.abs(joint.angle) + Math.abs(joint.velocity) * 0.12;
    }
    for (let j = 0; j < this.joints.length; j += 1) {
      adaptiveGain += this.adaptiveGain[j];
      trackingError += Math.abs(this.trackingError[j]);
    }
    for (let b = 0; b < this.boneMotorCounts.length; b += 1) {
      if (this.boneMotorCounts[b] > 0) coveredBones += 1;
    }
    return {
      motors: this.count,
      bones: this.boneCount,
      coveredBones,
      time: this.time,
      avgTemp: temp / this.count,
      avgCurrent: current / this.count,
      energy: this.energy,
      protectedRatio: protectedCount / this.count,
      maxAbsHeight: jointActivity,
      meanStrain: meanStrain / this.count,
      jointActivity,
      activeJoints: this.joints.filter((joint) => Math.abs(joint.angle) > 0.01).length,
      adaptiveGainMean: adaptiveGain / this.joints.length,
      trackingErrorMean: trackingError / this.joints.length,
      topControl: this.config.mode
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
        return this.deformation;
    }
  }

  pose() {
    const joint = (name) => this.joints[this.jointLookup.get(name)].angle;
    const root = { x: 0, y: 0.92 };
    const spineAngle = Math.PI / 2 + joint("spine");
    const torsoTop = add(root, vector(spineAngle, 0.54));
    const shoulderPerp = vector(spineAngle + Math.PI / 2, 0.19);
    const hipPerp = { x: 0.13, y: 0 };
    const leftShoulder = add(torsoTop, shoulderPerp);
    const rightShoulder = add(torsoTop, { x: -shoulderPerp.x, y: -shoulderPerp.y });
    const leftHip = add(root, { x: hipPerp.x, y: hipPerp.y });
    const rightHip = add(root, { x: -hipPerp.x, y: hipPerp.y });

    const lUpperArmAngle = -Math.PI / 2 + joint("lShoulder");
    const rUpperArmAngle = -Math.PI / 2 + joint("rShoulder");
    const lElbow = add(leftShoulder, vector(lUpperArmAngle, 0.32));
    const rElbow = add(rightShoulder, vector(rUpperArmAngle, 0.32));
    const lForearmAngle = lUpperArmAngle + joint("lElbow") * 0.72;
    const rForearmAngle = rUpperArmAngle + joint("rElbow") * 0.72;
    const lWrist = add(lElbow, vector(lForearmAngle, 0.30));
    const rWrist = add(rElbow, vector(rForearmAngle, 0.30));

    const lThighAngle = -Math.PI / 2 + joint("lHip");
    const rThighAngle = -Math.PI / 2 + joint("rHip");
    const lKnee = add(leftHip, vector(lThighAngle, 0.45));
    const rKnee = add(rightHip, vector(rThighAngle, 0.45));
    const lShinAngle = lThighAngle - joint("lKnee") * 0.62;
    const rShinAngle = rThighAngle - joint("rKnee") * 0.62;
    const lAnkle = add(lKnee, vector(lShinAngle, 0.43));
    const rAnkle = add(rKnee, vector(rShinAngle, 0.43));
    const lFootAngle = joint("lAnkle") * 0.45;
    const rFootAngle = joint("rAnkle") * 0.45;
    const neckAngle = spineAngle + joint("neck") * 0.5;

    const segments = {
      head: { start: torsoTop, angle: neckAngle, length: 0.20, width: 0.16 },
      torso: { start: torsoTop, angle: spineAngle + Math.PI, length: 0.54, width: 0.30 },
      pelvis: { start: { x: -0.16, y: root.y }, angle: 0, length: 0.32, width: 0.16 },
      lUpperArm: { start: leftShoulder, angle: lUpperArmAngle, length: 0.32, width: 0.09 },
      rUpperArm: { start: rightShoulder, angle: rUpperArmAngle, length: 0.32, width: 0.09 },
      lForearm: { start: lElbow, angle: lForearmAngle, length: 0.30, width: 0.075 },
      rForearm: { start: rElbow, angle: rForearmAngle, length: 0.30, width: 0.075 },
      lHand: { start: lWrist, angle: lForearmAngle, length: 0.10, width: 0.08 },
      rHand: { start: rWrist, angle: rForearmAngle, length: 0.10, width: 0.08 },
      lThigh: { start: leftHip, angle: lThighAngle, length: 0.45, width: 0.12 },
      rThigh: { start: rightHip, angle: rThighAngle, length: 0.45, width: 0.12 },
      lShin: { start: lKnee, angle: lShinAngle, length: 0.43, width: 0.095 },
      rShin: { start: rKnee, angle: rShinAngle, length: 0.43, width: 0.095 },
      lFoot: { start: lAnkle, angle: lFootAngle, length: 0.20, width: 0.08 },
      rFoot: { start: rAnkle, angle: Math.PI - rFootAngle, length: 0.20, width: 0.08 }
    };

    const nodes = {
      root,
      torsoTop,
      leftShoulder,
      rightShoulder,
      lElbow,
      rElbow,
      lWrist,
      rWrist,
      leftHip,
      rightHip,
      lKnee,
      rKnee,
      lAnkle,
      rAnkle
    };

    return { segments, nodes };
  }
}
