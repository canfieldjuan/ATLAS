/**
 * ╔══════════════════════════════════════════════════════╗
 * ║             ATLAS SCENE  —  scene.js                 ║
 * ║                                                      ║
 * ║  Sets up the robot and choreographs its sequence.    ║
 * ║                                                      ║
 * ║  HOW TO CUSTOMIZE:                                   ║
 * ║  • Change GROUND_Y to reposition the floor           ║
 * ║  • Change ROBOT_SCALE to resize the robot            ║
 * ║  • Edit the choreography() function to change moves  ║
 * ║  • Add new robot.xxx() calls from robot.js           ║
 * ╚══════════════════════════════════════════════════════╝
 */

// ─── Scene Constants ─────────────────────────────────────
// Adjust these to reposition elements in the 900×440 viewBox

const GROUND_Y     = 362;   // Y where the robot's feet sit in the scene
const ROBOT_SCALE  = 0.92;  // robot size (1 = 130px tall, 0.9 ≈ 117px)
const ROBOT_START  = -80;   // robot enters from off-screen left
const ROBOT_END    = 980;   // robot exits off-screen right (unused in loop)

// X positions for each letter in "ATLAS" (rough centers in 900px viewBox)
// Adjust if the layout changes.
const LETTERS = {
  A1: 192,   // first  A
  T:  316,   // T
  L:  430,   // L
  A2: 550,   // second A
  S:  672,   // S
};

// ─── Jump positions ──────────────────────────────────────
// Robot hops at these X coords as it crosses the word ATLAS.
// These are between / at letters for a "hopping over letters" feel.
const JUMP_POINTS = [
  { x: LETTERS.A1, height: 40, type: 0 },   // hop over A
  { x: LETTERS.T,  height: 48, type: 0 },   // hop over T
  { x: LETTERS.L,  height: 36, type: 0 },   // hop over L
  { x: LETTERS.A2, height: 44, type: 2 },   // wide legs over A
  { x: LETTERS.S,  height: 52, type: 1 },   // spin over S
];

// ─────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {

  const mount = document.getElementById('robot-mount');
  if (!mount) { console.error('robot-mount not found'); return; }

  // Create the robot
  const robot = new Robot(mount, {
    x:     ROBOT_START,
    y:     GROUND_Y,
    scale: ROBOT_SCALE,
  });

  // ── Background star twinkle ─────────────────────────
  startStarTwinkle();

  // ── Start main choreography loop ───────────────────
  runChoreography(robot);
});


/**
 * Main choreography sequence.
 * Runs then repeats from scratch via recursion.
 */
function runChoreography(robot) {
  const tl = gsap.timeline({ onComplete: () => runChoreography(robot) });

  // ── 1. Enter from left ─────────────────────────────
  tl.add(robot.walk(LETTERS.A1 - 60, 130));   // jog to just left of ATLAS

  // ── 2. Look at the logo ────────────────────────────
  tl.add(robot.look('right'));
  tl.add(robot.think(0.8));                    // thinking pose

  // ── 3. Walk across ATLAS, jumping at each letter ──
  tl.add(buildCrossing(robot));

  // ── 4. Pause at end, turn around ──────────────────
  tl.add(robot.look('reset'));
  tl.add(robot.turn(-1));                      // flip to face left

  // ── 5. Walk back to center ─────────────────────────
  tl.add(robot.walk(450, 100));                // stroll back to middle

  // ── 6. Dance ──────────────────────────────────────
  tl.add(robot.dance());
  tl.add(robot.dance());

  // ── 7. Victory then sit ────────────────────────────
  tl.add(robot.victory());
  tl.add(robot.sit(1.0));                      // sit and rest 1 second

  // ── 8. Stand up, wave, idle ────────────────────────
  tl.add(robot.turn(1));                       // face right again
  tl.add(robot.wave(3));                       // big wave
  tl.add(robot.reset());
  tl.add(idlePause(robot, 1.6));              // breathe for 1.6s

  // ── 9. Walk offscreen left for a fresh loop entry ─
  tl.add(robot.turn(-1));
  tl.add(robot.walk(ROBOT_START, 160));        // fast exit

  return tl;
}


/**
 * Build the "robot crosses ATLAS" sub-sequence:
 * walks between each letter, jumps at each one.
 */
function buildCrossing(robot) {
  const tl = gsap.timeline();

  // Walk to each jump point and hop
  let prevX = robot.x;

  JUMP_POINTS.forEach(({ x, height, type }) => {
    // Walk up to the letter
    tl.add(robot.walk(x, 105));
    // Jump!
    tl.add(robot.jump(height, type));
    prevX = x;
  });

  // Walk past the last letter to right edge
  tl.add(robot.walk(LETTERS.S + 130, 105));

  return tl;
}


/**
 * Hold idle animation for `duration` seconds.
 * Idle starts WHEN this point in the sequence is reached (via callback),
 * not when the file loads. Returns a timeline so it can be chained.
 */
function idlePause(robot, duration) {
  const tl = gsap.timeline();
  let idleTl;
  // Start idle at the moment this timeline begins (not at build time)
  tl.add(() => { idleTl = robot.idle(); });
  tl.to({}, { duration });
  tl.add(() => { if (idleTl) idleTl.kill(); });
  tl.add(robot.reset(0.25));
  return tl;
}


/**
 * Make background stars slowly twinkle (opacity pulse).
 */
function startStarTwinkle() {
  const stars = document.querySelectorAll('#bg-stars circle, g[opacity="0.45"] circle');
  stars.forEach((star, i) => {
    gsap.to(star, {
      opacity: Math.random() * 0.4 + 0.1,
      duration: Math.random() * 2 + 1.5,
      repeat:   -1,
      yoyo:      true,
      delay:     i * 0.15,
      ease:      'sine.inOut',
    });
  });
}
