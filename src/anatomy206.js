const SIDES = ["left", "right"];

function addBone(bones, name, group, segment, joint, weight = 1, side = "midline") {
  bones.push({
    id: bones.length,
    name,
    group,
    segment,
    joint,
    weight,
    side
  });
}

function addPair(bones, baseName, group, segmentForSide, jointForSide, weight = 1) {
  for (const side of SIDES) {
    const segment = typeof segmentForSide === "function" ? segmentForSide(side) : segmentForSide;
    const joint = typeof jointForSide === "function" ? jointForSide(side) : jointForSide;
    addBone(bones, `${side} ${baseName}`, group, segment, joint, weight, side);
  }
}

function addSeries(bones, prefix, count, group, segment, joint, weight = 1) {
  for (let i = 1; i <= count; i += 1) {
    addBone(bones, `${prefix}${i}`, group, segment, joint, weight);
  }
}

function addHandBones(bones, side) {
  const segment = side === "left" ? "lHand" : "rHand";
  const joint = side === "left" ? "lElbow" : "rElbow";
  const carpals = ["scaphoid", "lunate", "triquetrum", "pisiform", "trapezium", "trapezoid", "capitate", "hamate"];
  for (const carpal of carpals) addBone(bones, `${side} ${carpal}`, "carpal", segment, joint, 0.55, side);
  for (let i = 1; i <= 5; i += 1) addBone(bones, `${side} metacarpal ${i}`, "metacarpal", segment, joint, 0.7, side);
  addBone(bones, `${side} thumb proximal phalanx`, "hand phalanx", segment, joint, 0.45, side);
  addBone(bones, `${side} thumb distal phalanx`, "hand phalanx", segment, joint, 0.4, side);
  for (const finger of ["index", "middle", "ring", "little"]) {
    for (const part of ["proximal", "middle", "distal"]) {
      addBone(bones, `${side} ${finger} ${part} phalanx`, "hand phalanx", segment, joint, 0.42, side);
    }
  }
}

function addFootBones(bones, side) {
  const segment = side === "left" ? "lFoot" : "rFoot";
  const joint = side === "left" ? "lAnkle" : "rAnkle";
  const tarsals = ["talus", "calcaneus", "navicular", "cuboid", "medial cuneiform", "intermediate cuneiform", "lateral cuneiform"];
  for (const tarsal of tarsals) addBone(bones, `${side} ${tarsal}`, "tarsal", segment, joint, 0.72, side);
  for (let i = 1; i <= 5; i += 1) addBone(bones, `${side} metatarsal ${i}`, "metatarsal", segment, joint, 0.65, side);
  addBone(bones, `${side} hallux proximal phalanx`, "foot phalanx", segment, joint, 0.42, side);
  addBone(bones, `${side} hallux distal phalanx`, "foot phalanx", segment, joint, 0.38, side);
  for (const toe of ["second", "third", "fourth", "fifth"]) {
    for (const part of ["proximal", "middle", "distal"]) {
      addBone(bones, `${side} ${toe} toe ${part} phalanx`, "foot phalanx", segment, joint, 0.36, side);
    }
  }
}

function buildAnatomy206() {
  const bones = [];

  for (const name of ["frontal", "occipital", "sphenoid", "ethmoid"]) addBone(bones, name, "cranial", "head", "neck", 1.2);
  addPair(bones, "parietal", "cranial", "head", "neck", 1.0);
  addPair(bones, "temporal", "cranial", "head", "neck", 0.95);
  addBone(bones, "mandible", "facial", "head", "neck", 0.9);
  addBone(bones, "vomer", "facial", "head", "neck", 0.25);
  for (const paired of ["maxilla", "zygomatic", "nasal", "lacrimal", "palatine", "inferior nasal concha"]) {
    addPair(bones, paired, "facial", "head", "neck", paired === "maxilla" ? 0.75 : 0.32);
  }
  for (const ossicle of ["malleus", "incus", "stapes"]) addPair(bones, ossicle, "auditory ossicle", "head", "neck", 0.08);
  addBone(bones, "hyoid", "hyoid", "head", "neck", 0.18);

  addSeries(bones, "C", 7, "cervical vertebra", "torso", "neck", 0.7);
  addSeries(bones, "T", 12, "thoracic vertebra", "torso", "spine", 0.9);
  addSeries(bones, "L", 5, "lumbar vertebra", "torso", "spine", 1.1);
  addBone(bones, "sacrum", "vertebral", "pelvis", "spine", 1.4);
  addBone(bones, "coccyx", "vertebral", "pelvis", "spine", 0.45);
  addBone(bones, "sternum", "thoracic cage", "torso", "spine", 1.0);
  for (let i = 1; i <= 12; i += 1) {
    addPair(bones, `rib ${i}`, "rib", "torso", "spine", i <= 7 ? 0.78 : 0.55);
  }

  addPair(bones, "clavicle", "pectoral girdle", (side) => side === "left" ? "lUpperArm" : "rUpperArm", (side) => side === "left" ? "lShoulder" : "rShoulder", 0.85);
  addPair(bones, "scapula", "pectoral girdle", (side) => side === "left" ? "lUpperArm" : "rUpperArm", (side) => side === "left" ? "lShoulder" : "rShoulder", 1.15);
  addPair(bones, "humerus", "upper limb", (side) => side === "left" ? "lUpperArm" : "rUpperArm", (side) => side === "left" ? "lShoulder" : "rShoulder", 1.35);
  addPair(bones, "radius", "upper limb", (side) => side === "left" ? "lForearm" : "rForearm", (side) => side === "left" ? "lElbow" : "rElbow", 1.0);
  addPair(bones, "ulna", "upper limb", (side) => side === "left" ? "lForearm" : "rForearm", (side) => side === "left" ? "lElbow" : "rElbow", 1.0);
  for (const side of SIDES) addHandBones(bones, side);

  addPair(bones, "hip bone", "pelvic girdle", (side) => side === "left" ? "lThigh" : "rThigh", (side) => side === "left" ? "lHip" : "rHip", 1.45);
  addPair(bones, "femur", "lower limb", (side) => side === "left" ? "lThigh" : "rThigh", (side) => side === "left" ? "lHip" : "rHip", 1.65);
  addPair(bones, "patella", "lower limb", (side) => side === "left" ? "lThigh" : "rThigh", (side) => side === "left" ? "lKnee" : "rKnee", 0.55);
  addPair(bones, "tibia", "lower limb", (side) => side === "left" ? "lShin" : "rShin", (side) => side === "left" ? "lKnee" : "rKnee", 1.35);
  addPair(bones, "fibula", "lower limb", (side) => side === "left" ? "lShin" : "rShin", (side) => side === "left" ? "lKnee" : "rKnee", 0.85);
  for (const side of SIDES) addFootBones(bones, side);

  if (bones.length !== 206) {
    throw new Error(`Anatomy catalog expected 206 bones, found ${bones.length}`);
  }
  return bones;
}

export const ANATOMY_206 = Object.freeze(buildAnatomy206());

export function anatomySummary(bones = ANATOMY_206) {
  const byGroup = new Map();
  const bySegment = new Map();
  for (const bone of bones) {
    byGroup.set(bone.group, (byGroup.get(bone.group) || 0) + 1);
    bySegment.set(bone.segment, (bySegment.get(bone.segment) || 0) + 1);
  }
  return {
    totalBones: bones.length,
    groups: Object.fromEntries([...byGroup.entries()].sort()),
    segments: Object.fromEntries([...bySegment.entries()].sort())
  };
}
