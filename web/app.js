/**
 * SAM-Audio Browser Application
 *
 * Orchestrates ONNX Runtime Web sessions for audio source separation.
 * Pipeline: audio → DACVAE encode → T5 encode text → ODE loop (DiT × 32) → DACVAE decode
 */

import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.2/dist/transformers.min.js";

// Constants
const SAMPLE_RATE = 48000;
const HOP_LENGTH = 1920; // 2 * 8 * 10 * 12
const ODE_STEPS = 16; // midpoint method: 16 steps = 32 function evaluations
const ODE_DT = 2.0 / 32;
const VISION_DIM = 1024;

// State
let sessions = {};
let tokenizer = null;
let modelsLoaded = false;

// --- Logging ---

function log(msg) {
  const el = document.getElementById("log");
  const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
  el.textContent += `[${ts}] ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}

// --- Model Loading ---

async function createSession(url, name, progressCb) {
  log(`Loading ${name}...`);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const total = parseInt(resp.headers.get("content-length") || "0", 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (total > 0 && progressCb) progressCb(loaded / total);
  }
  const buf = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.byteLength;
  }

  const providers = [];
  if (ort.env.webgpu && typeof navigator !== "undefined" && navigator.gpu) {
    providers.push("webgpu");
  }
  providers.push("wasm");

  const session = await ort.InferenceSession.create(buf.buffer, {
    executionProviders: providers,
  });
  log(`  ${name} loaded (${(loaded / 1024 / 1024).toFixed(1)} MB) [${providers[0]}]`);
  return session;
}

window.loadModels = async function () {
  const btn = document.getElementById("btn-load");
  const progress = document.getElementById("load-progress");
  const status = document.getElementById("load-status");
  btn.disabled = true;
  status.textContent = "Loading...";
  status.classList.remove("error");

  let baseUrl = document.getElementById("model-url").value.trim();
  if (!baseUrl.endsWith("/")) baseUrl += "/";

  const files = [
    ["dacvae_encoder", "dacvae_encoder.onnx"],
    ["dacvae_decoder", "dacvae_decoder.onnx"],
    ["t5_encoder", "t5_encoder.onnx"],
    ["dit_forward", "dit_forward.onnx"],
  ];

  try {
    let fileIdx = 0;
    for (const [key, filename] of files) {
      const i = fileIdx++;
      sessions[key] = await createSession(
        baseUrl + filename,
        filename,
        (p) => {
          const overall = (i + p) / files.length;
          progress.style.width = `${overall * 100}%`;
        }
      );
    }

    log("Loading T5 tokenizer...");
    tokenizer = await AutoTokenizer.from_pretrained("Xenova/t5-base");
    log("  T5 tokenizer loaded");

    progress.style.width = "100%";
    status.textContent = "Models loaded. Ready to separate audio.";
    modelsLoaded = true;
    document.getElementById("btn-separate").disabled = false;
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
    status.classList.add("error");
    log(`ERROR: ${e.message}`);
    console.error(e);
  } finally {
    btn.disabled = false;
  }
};

// --- Audio Processing ---

async function loadAudioFile(file) {
  const arrayBuf = await file.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: SAMPLE_RATE,
  });
  const decoded = await audioCtx.decodeAudioData(arrayBuf);

  // Mix to mono
  const numSamples = decoded.length;
  const mono = new Float32Array(numSamples);
  for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
    const channelData = decoded.getChannelData(ch);
    for (let i = 0; i < numSamples; i++) {
      mono[i] += channelData[i] / decoded.numberOfChannels;
    }
  }

  // Pad to hop_length multiple
  const padded = padToHop(mono);
  log(`Audio: ${(padded.length / SAMPLE_RATE).toFixed(1)}s, ${padded.length} samples (${decoded.numberOfChannels}ch -> mono, ${decoded.sampleRate}Hz -> ${SAMPLE_RATE}Hz)`);
  audioCtx.close();
  return padded;
}

function padToHop(samples) {
  const remainder = samples.length % HOP_LENGTH;
  if (remainder === 0) return samples;
  const padded = new Float32Array(samples.length + HOP_LENGTH - remainder);
  padded.set(samples);
  return padded;
}

function createWavBlob(samples, sampleRate) {
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  // WAV header
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);

  // Convert float32 to int16
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

// --- Tensor Helpers ---

function randn(shape) {
  // Box-Muller transform for normal random numbers
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.random();
    const u2 = Math.random();
    const r = Math.sqrt(-2 * Math.log(u1));
    data[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < size) data[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return new ort.Tensor("float32", data, shape);
}

function tensorAdd(a, b) {
  const result = new Float32Array(a.data.length);
  for (let i = 0; i < result.length; i++) result[i] = a.data[i] + b.data[i];
  return new ort.Tensor(a.type, result, a.dims);
}

function tensorScale(a, s) {
  const result = new Float32Array(a.data.length);
  for (let i = 0; i < result.length; i++) result[i] = a.data[i] * s;
  return new ort.Tensor(a.type, result, a.dims);
}

function tensorZeros(shape, type = "float32") {
  return new ort.Tensor(type, new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
}

function tensorOnes(shape, type = "bool") {
  const size = shape.reduce((a, b) => a * b, 1);
  if (type === "bool") {
    const data = new Uint8Array(size).fill(1);
    return new ort.Tensor(type, data, shape);
  }
  const data = new Float32Array(size).fill(1);
  return new ort.Tensor("float32", data, shape);
}

// --- Inference Pipeline ---

async function encodeAudio(waveform) {
  // waveform: Float32Array [samples] -> tensor [1, 1, samples]
  const input = new ort.Tensor("float32", waveform, [1, 1, waveform.length]);
  const result = await sessions.dacvae_encoder.run({ waveform: input });
  return result.features; // [1, C, T]
}

async function encodeText(text) {
  const encoded = await tokenizer(text, {
    padding: true,
    truncation: true,
    max_length: 512,
    return_tensors: "js",
  });

  const inputIds = new ort.Tensor("int64", BigInt64Array.from(encoded.input_ids.data.map(BigInt)), encoded.input_ids.dims);
  const attentionMask = new ort.Tensor("int64", BigInt64Array.from(encoded.attention_mask.data.map(BigInt)), encoded.attention_mask.dims);

  const result = await sessions.t5_encoder.run({
    input_ids: inputIds,
    attention_mask: attentionMask,
  });

  // Convert attention_mask to bool for DiT
  const maskBool = new Uint8Array(encoded.attention_mask.data.length);
  for (let i = 0; i < maskBool.length; i++) maskBool[i] = encoded.attention_mask.data[i] ? 1 : 0;
  const textMask = new ort.Tensor("bool", maskBool, encoded.attention_mask.dims);

  return { textFeatures: result.last_hidden_state, textMask };
}

async function odeSolve(noise, audioFeatures, textFeatures, textMask, maskedVideoFeatures, audioPadMask, onStep) {
  let y = noise;

  for (let step = 0; step < ODE_STEPS; step++) {
    const tVal = step * ODE_DT;

    // k1 = f(t, y)
    const t1 = new ort.Tensor("float32", new Float32Array([tVal]), [1]);
    const res1 = await sessions.dit_forward.run({
      noisy_audio: y,
      audio_features: audioFeatures,
      text_features: textFeatures,
      time: t1,
      masked_video_features: maskedVideoFeatures,
      text_mask: textMask,
      audio_pad_mask: audioPadMask,
    });
    const k1 = res1.velocity;

    // y_mid = y + k1 * (dt/2)
    const yMid = tensorAdd(y, tensorScale(k1, ODE_DT / 2));

    // k2 = f(t + dt/2, y_mid)
    const t2 = new ort.Tensor("float32", new Float32Array([tVal + ODE_DT / 2]), [1]);
    const res2 = await sessions.dit_forward.run({
      noisy_audio: yMid,
      audio_features: audioFeatures,
      text_features: textFeatures,
      time: t2,
      masked_video_features: maskedVideoFeatures,
      text_mask: textMask,
      audio_pad_mask: audioPadMask,
    });
    const k2 = res2.velocity;

    // y = y + k2 * dt
    y = tensorAdd(y, tensorScale(k2, ODE_DT));

    if (onStep) onStep(step + 1, ODE_STEPS);
  }

  return y;
}

async function decodeAudio(features) {
  const result = await sessions.dacvae_decoder.run({ features });
  return result.waveform; // [1, 1, samples]
}

window.separate = async function () {
  if (!modelsLoaded) return;

  const btn = document.getElementById("btn-separate");
  const progress = document.getElementById("sep-progress");
  const status = document.getElementById("sep-status");
  btn.disabled = true;
  status.textContent = "Processing...";
  status.classList.remove("error");
  progress.style.width = "0%";

  try {
    // Load audio
    const fileInput = document.getElementById("audio-file");
    if (!fileInput.files.length) throw new Error("Please select an audio file");
    const text = document.getElementById("text-prompt").value.trim();
    if (!text) throw new Error("Please enter a text prompt");

    log("--- Starting separation ---");
    const t0 = performance.now();

    status.textContent = "Loading audio...";
    const waveform = await loadAudioFile(fileInput.files[0]);

    // 1. Encode audio
    status.textContent = "Encoding audio (DACVAE)...";
    log("Encoding audio...");
    let t1 = performance.now();
    const features = await encodeAudio(waveform); // [1, C, T]
    log(`  DACVAE encode: ${((performance.now() - t1) / 1000).toFixed(1)}s`);
    progress.style.width = "5%";

    // Prepare audio features: transpose [1,C,T] -> [1,T,C], then concat [1,T,2C]
    const [B, C, T] = features.dims;
    const transposed = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < C; c++) {
        transposed[t * C + c] = features.data[c * T + t];
      }
    }
    const doubled = new Float32Array(T * 2 * C);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < C; c++) {
        doubled[t * 2 * C + c] = transposed[t * C + c];
        doubled[t * 2 * C + C + c] = transposed[t * C + c];
      }
    }
    const audioFeatures = new ort.Tensor("float32", doubled, [1, T, 2 * C]);

    // 2. Encode text
    status.textContent = "Encoding text (T5)...";
    log(`Encoding text: "${text}"`);
    t1 = performance.now();
    const { textFeatures, textMask } = await encodeText(text);
    log(`  T5 encode: ${((performance.now() - t1) / 1000).toFixed(1)}s`);
    progress.style.width = "10%";

    // 3. Prepare remaining inputs
    const maskedVideoFeatures = tensorZeros([1, VISION_DIM, T]);
    const audioPadMask = tensorOnes([1, T], "bool");
    const noise = randn([1, T, 2 * C]);

    // 4. ODE solve
    status.textContent = "Running diffusion (0/16 steps)...";
    log("ODE solve (midpoint, 16 steps)...");
    t1 = performance.now();
    const generated = await odeSolve(
      noise, audioFeatures, textFeatures, textMask,
      maskedVideoFeatures, audioPadMask,
      (step, total) => {
        const pct = 10 + (step / total) * 80;
        progress.style.width = `${pct}%`;
        status.textContent = `Running diffusion (${step}/${total} steps)...`;
      }
    );
    log(`  ODE solve: ${((performance.now() - t1) / 1000).toFixed(1)}s`);

    // 5. Decode: generated is [1, T, 2C]. Split into target [1,C,T] and residual [1,C,T]
    status.textContent = "Decoding audio (DACVAE)...";
    log("Decoding audio...");
    t1 = performance.now();

    // Transpose [1,T,2C] -> split target/residual -> [1,C,T] each
    const targetData = new Float32Array(C * T);
    const residualData = new Float32Array(C * T);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < C; c++) {
        targetData[c * T + t] = generated.data[t * 2 * C + c];
        residualData[c * T + t] = generated.data[t * 2 * C + C + c];
      }
    }

    const targetFeatures = new ort.Tensor("float32", targetData, [1, C, T]);
    const residualFeatures = new ort.Tensor("float32", residualData, [1, C, T]);

    const targetWav = await decodeAudio(targetFeatures);
    const residualWav = await decodeAudio(residualFeatures);
    log(`  DACVAE decode: ${((performance.now() - t1) / 1000).toFixed(1)}s`);
    progress.style.width = "100%";

    // Trim to original length
    const origSamples = waveform.length;
    const targetSamples = new Float32Array(targetWav.data.buffer, 0, Math.min(targetWav.data.length, origSamples));
    const residualSamples = new Float32Array(residualWav.data.buffer, 0, Math.min(residualWav.data.length, origSamples));

    // Create audio blobs and display
    const targetBlob = createWavBlob(targetSamples, SAMPLE_RATE);
    const residualBlob = createWavBlob(residualSamples, SAMPLE_RATE);

    document.getElementById("audio-target").src = URL.createObjectURL(targetBlob);
    document.getElementById("audio-residual").src = URL.createObjectURL(residualBlob);
    document.getElementById("results-section").classList.remove("hidden");

    const totalTime = (performance.now() - t0) / 1000;
    status.textContent = `Done in ${totalTime.toFixed(1)}s`;
    log(`--- Separation complete: ${totalTime.toFixed(1)}s total ---`);

  } catch (e) {
    status.textContent = `Error: ${e.message}`;
    status.classList.add("error");
    log(`ERROR: ${e.message}`);
    console.error(e);
  } finally {
    btn.disabled = false;
  }
};

// --- File input preview ---

document.getElementById("audio-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const url = URL.createObjectURL(file);
    document.getElementById("input-preview").src = url;
    document.getElementById("input-preview-row").classList.remove("hidden");
  }
});
