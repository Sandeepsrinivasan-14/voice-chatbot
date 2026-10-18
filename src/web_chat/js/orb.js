/* The voice orb: a single full-canvas fragment shader (plain WebGL, no
 * library, so there is nothing to vendor and nothing to fetch -- the app is
 * fully offline). It is lazy-loaded by main.js after first paint.
 *
 * The sphere's silhouette and surface are driven by 3D simplex noise whose
 * amplitude follows the live audio level; colours cross-fade per phase
 * (idle / listening / thinking / speaking); two orbital rings spin up while
 * the assistant thinks. If WebGL is unavailable this module reports failure
 * and the CSS orb in app.css stays on screen.
 *
 * Simplex noise: Ashima Arts / Stefan Gustavson, MIT licence.
 */

const VERT = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG = `
precision highp float;
varying vec2 vUv;
uniform vec2 uRes;
uniform float uTime;
uniform float uLevel;
uniform vec3 uBands;   // bass, mid, treble
uniform vec4 uPh;      // phase weights: idle, listening, thinking, speaking
uniform vec3 uA;
uniform vec3 uB;
uniform vec3 uC;
uniform vec2 uTilt;
uniform float uHalo;

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(
            i.z + vec4(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
          + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

mat2 rot(float a) { float c = cos(a); float s = sin(a); return mat2(c, -s, s, c); }

// A thin orbital ring seen at an angle. Returns (alpha, isFront).
vec2 orbitRing(vec2 p, float rad, float tilt, float rotA, float spin, float px) {
  vec2 q = rot(rotA) * p;
  float b = rad * cos(tilt);
  vec2 e = vec2(q.x / rad, q.y / b);
  float dist = abs(length(e) - 1.0) * min(rad, b);
  float line = 1.0 - smoothstep(0.0, px * 1.4, dist);
  float phi = atan(e.y, e.x);
  float behind = mod(spin - phi, 6.2831853);        // 0 at the comet head, grows along the tail
  float comet = exp(-behind * 0.85) + 0.18;
  float front = step(e.y, 0.0);                      // lower half of the ellipse is nearer the viewer
  return vec2(line * comet, front);
}

void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float px = 2.0 / uRes.y;
  float r = length(p);
  float ang = atan(p.y, p.x);
  float t = uTime;

  float reactive = uPh.y * 0.95 + uPh.w;
  float energy = clamp(uLevel * reactive + 0.06 * uPh.z + 0.03, 0.0, 1.2);

  // ---- silhouette: base radius deformed by noise that grows with energy ----
  vec2 dir = vec2(cos(ang), sin(ang));
  float n1 = snoise(vec3(dir * 1.15, t * 0.35 + uBands.x * 1.5));
  float n2 = snoise(vec3(dir * 2.7, t * 0.95 + 7.0));
  float R = 0.63
          + 0.013 * sin(t * 1.4) * uPh.x
          + 0.04 * uLevel * reactive
          + (0.012 + 0.12 * energy) * n1
          + 0.04 * uBands.z * reactive * n2;
  R = clamp(R, 0.3, 0.8);
  float inside = 1.0 - smoothstep(R - px * 1.2, R + px * 1.2, r);

  // ---- surface ----
  float zc = sqrt(max(1.0 - (r * r) / (R * R), 0.0));
  vec3 n = normalize(vec3(p / R, zc));
  vec3 q = n;
  q.xz = rot(uTilt.x * 0.9 + t * (0.12 + 0.6 * uPh.z)) * q.xz;
  q.yz = rot(uTilt.y * 0.9) * q.yz;
  float flow = t * 0.25 + uLevel * 1.2 * reactive + t * 0.7 * uPh.z;
  float s1 = snoise(q * 1.7 + vec3(0.0, flow, 0.0));
  float s2 = snoise(q * 2.9 + vec3(s1 * 0.6) - vec3(flow * 0.8, 0.0, 3.0));
  float s3 = snoise(q * 2.0 + vec3(11.0, 0.0, flow * 0.4));
  float ridge = 1.0 - abs(s2);

  vec3 col = mix(uA, uB, smoothstep(-0.35, 0.45, s1));
  col = mix(col, uC, pow(ridge, 3.0) * (0.3 + 0.5 * energy));

  vec3 pn = normalize(n + 0.24 * vec3(s1, s2, s3) * (0.5 + energy));
  vec3 L = normalize(vec3(-0.45, 0.65, 0.62));
  float diff = clamp(dot(pn, L) * 0.5 + 0.5, 0.0, 1.0);
  float fres = pow(1.0 - zc, 2.2);
  float spec = pow(max(dot(reflect(-L, pn), vec3(0.0, 0.0, 1.0)), 0.0), 36.0);

  col *= mix(0.5, 1.15, diff);
  col += uC * fres * (0.7 + 1.1 * energy);
  col += vec3(1.0) * spec * 0.55;
  col += uB * pow(zc, 3.0) * (0.1 + 0.35 * energy);

  // ---- halo + sound ripples outside the body ----
  float d = max(r - R, 0.0);
  float halo = exp(-d * 4.4) * (0.2 + 0.55 * energy + 0.1 * uPh.z) * uHalo;
  float ripple = (0.5 + 0.5 * sin(d * 34.0 - t * 3.2)) * exp(-d * 5.5) * uLevel * reactive * 0.5 * uHalo;
  vec3 haloCol = mix(uB, uC, 0.5 + 0.5 * sin(ang * 2.0 + t * 0.6));
  float edgeFade = 1.0 - smoothstep(0.72, 1.0, r);   // never let the halo hit the canvas edge
  vec3 glow = haloCol * (halo + ripple) * (1.0 - inside) * edgeFade;

  // ---- orbital rings (visible while thinking, whisper-faint otherwise) ----
  float ringVis = 0.1 + 0.9 * uPh.z + 0.18 * energy;
  float spinA = t * (0.9 + 3.2 * uPh.z);
  vec2 r1 = orbitRing(p, 0.92, 1.15, 0.5 + t * 0.12, spinA, px);
  vec2 r2 = orbitRing(p, 0.8, 1.0, -0.9 - t * 0.09, -spinA * 0.8 + 3.0, px);
  float vis1 = r1.y > 0.5 ? 1.0 : (1.0 - inside);
  float vis2 = r2.y > 0.5 ? 1.0 : (1.0 - inside);
  vec3 ringCol = (uC * r1.x * vis1 + uB * r2.x * vis2) * ringVis * 0.85;

  vec3 rgb = col * inside + glow + ringCol * edgeFade;
  float a = clamp(max(inside, max(max(rgb.r, rgb.g), rgb.b)), 0.0, 1.0);
  rgb = min(rgb, vec3(a));
  gl_FragColor = vec4(rgb, a);          // premultiplied alpha
}`;

const UNIFORMS = ["uRes", "uTime", "uLevel", "uBands", "uPh", "uA", "uB", "uC", "uTilt", "uHalo"];

function compile(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error("shader compile failed: " + log);
  }
  return shader;
}

/* Resolves to the orb controller, or null if WebGL is unavailable. */
export function createOrb(canvas) {
  const gl =
    canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true, antialias: false, powerPreference: "low-power" }) ||
    canvas.getContext("experimental-webgl");
  if (!gl) return null;

  let program = null;
  let uniforms = {};
  let buffer = null;
  let lost = false;
  let scale = 1; // adaptive resolution scale
  let slowFrames = 0;

  function build() {
    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error("program link failed: " + gl.getProgramInfoLog(program));
    }
    gl.useProgram(program);
    buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(program, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    uniforms = {};
    for (const name of UNIFORMS) uniforms[name] = gl.getUniformLocation(program, name);
    gl.clearColor(0, 0, 0, 0);
  }

  try {
    build();
  } catch (err) {
    console.warn("[orb] WebGL orb disabled:", err.message);
    return null;
  }

  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    lost = true;
  });
  canvas.addEventListener("webglcontextrestored", () => {
    try {
      build();
      lost = false;
      resize();
    } catch (err) {
      lost = true;
    }
  });

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    const w = Math.max(2, Math.round(canvas.clientWidth * dpr * scale));
    const h = Math.max(2, Math.round(canvas.clientHeight * dpr * scale));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  resize();

  /* s: { time, level, bands:[b,m,t], phases:[i,l,th,sp], colors:[a,b,c], tilt:[x,y], halo, frameMs } */
  function render(s) {
    if (lost || canvas.clientWidth === 0) return;
    // Back off resolution if the device cannot hold ~30fps; the orb is soft
    // enough that a lower render scale is barely visible.
    if (s.frameMs > 34) slowFrames++;
    else slowFrames = Math.max(0, slowFrames - 1);
    if (slowFrames > 40 && scale > 0.5) {
      scale = Math.max(0.5, scale * 0.8);
      slowFrames = 0;
      resize();
    }
    gl.uniform2f(uniforms.uRes, canvas.width, canvas.height);
    gl.uniform1f(uniforms.uTime, s.time);
    gl.uniform1f(uniforms.uLevel, s.level);
    gl.uniform3f(uniforms.uBands, s.bands[0], s.bands[1], s.bands[2]);
    gl.uniform4f(uniforms.uPh, s.phases[0], s.phases[1], s.phases[2], s.phases[3]);
    gl.uniform3f(uniforms.uA, s.colors[0][0], s.colors[0][1], s.colors[0][2]);
    gl.uniform3f(uniforms.uB, s.colors[1][0], s.colors[1][1], s.colors[1][2]);
    gl.uniform3f(uniforms.uC, s.colors[2][0], s.colors[2][1], s.colors[2][2]);
    gl.uniform2f(uniforms.uTilt, s.tilt[0], s.tilt[1]);
    gl.uniform1f(uniforms.uHalo, s.halo);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  return { render, resize, get lost() { return lost; } };
}
