//! The Phonon TTS engine (Gradium Phonon 31M, English) — 5.6.0's second
//! engine, selected via Settings' `tts_engine: "phonon"`. Generates 24 kHz
//! mono through the same `on_chunk` contract as Kokoro (chunked so playback
//! can start before generation finishes), with the model files managed by
//! Settings → Voice Models like every other download (see `model.rs`).
//!
//! The Synth pipeline stays resident across calls (it is built once, behind
//! a mutex — Phonon's runtime is single-threaded per instance).

use anyhow::{anyhow, Context, Result};
use ptts::synth::Synth;
use std::sync::Mutex;
use tauri::{AppHandle, Wry};

pub const SAMPLE_RATE: u32 = 24_000;
const DEFAULT_VOICE: &str = "default";

static SYNTH: std::sync::OnceLock<Mutex<Option<Synth>>> = std::sync::OnceLock::new();

fn engine(app: &AppHandle<Wry>) -> Result<&'static Mutex<Option<Synth>>> {
    let cell = SYNTH.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().map_err(|_| anyhow!("Phonon lock poisoned"))?;
    if guard.is_none() {
        *guard = Some(build(app).map_err(|e| {
            anyhow!("Phonon engine failed to load: {e}")
        })?);
    }
    Ok(cell)
}

fn build(app: &AppHandle<Wry>) -> Result<Synth> {
    let (config, weights, voice, tokenizer) = crate::model::phonon_paths(app)?;
    for f in [&config, &weights, &voice, &tokenizer] {
        if !f.is_file() {
            return Err(anyhow!(
                "Phonon model not downloaded — add it in Settings → Voice Models"
            ));
        }
    }
    let cfg = ptts::checkpoint::read_config(config.to_string_lossy().as_ref())
        .context("read Phonon config.json")?;
    Synth::builder(cfg, weights.to_string_lossy().as_ref(), ptts::preprocess::Lang::En)
        .tokenizer_file(tokenizer.to_string_lossy().as_ref())
        .add_voice(DEFAULT_VOICE, voice.to_string_lossy().as_ref())
        .build()
        .map_err(|e| anyhow!("build Phonon Synth from {}: {e:#}", weights.display()))
}

/// Warm the engine at startup (called from `kokoro::preload` when Phonon is
/// the selected engine): builds the Synth and runs a one-word synthesis so
/// the first real speak pays neither the model load nor the kernel warm-up.
pub fn preload(app: AppHandle<Wry>) {
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        match speak_chunks(&app, "Ready.", "default", |_| Ok(true)) {
            Ok(()) => eprintln!("phonon: preloaded in {:?}", t0.elapsed()),
            Err(e) => eprintln!("phonon: preload failed: {e}"),
        }
    });
}

/// Split `text` into sentence-ish chunks (same idea as Kokoro's chunking —
/// Phonon generates whole-call-at-once otherwise, so the first audio waits
/// for the entire text to finish generating).
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for piece in text.split_inclusive(['.', '!', '?', ',', ';', ':']) {
        current.push_str(piece);
        if current.trim().len() >= 40 || piece.ends_with(['.', '!', '?']) {
            let c = std::mem::take(&mut current).trim().to_string();
            if !c.is_empty() {
                out.push(c);
            }
        }
    }
    let rest = std::mem::take(&mut current).trim().to_string();
    if !rest.is_empty() {
        out.push(rest);
    }
    if out.is_empty() {
        out.push(text.trim().to_string());
    }
    out
}

/// Synthesize `text` with Phonon sentence by sentence (each `say` streams
/// its ~1 s of audio straight to `on_chunk`; `Ok(false)` stops early).
/// `voice` is ignored — Phonon 31M ships a single default voice.
pub fn speak_chunks(
    app: &AppHandle<Wry>,
    text: &str,
    _voice: &str,
    mut on_chunk: impl FnMut(Vec<f32>) -> Result<bool>,
) -> Result<()> {
    if text.trim().is_empty() {
        return Err(anyhow!("nothing to speak"));
    }
    let cell = engine(app)?;
    let mut produced = false;
    for sentence in sentences(text) {
        let mut guard = cell.lock().map_err(|_| anyhow!("Phonon lock poisoned"))?;
        let tts = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Phonon engine not initialized"))?;
        let pcm = tts
            .say(&sentence)
            .map_err(|e| anyhow!("Phonon synthesis failed: {e}"))?;
        let rate = tts.sample_rate() as usize;
        drop(guard);
        if pcm.is_empty() {
            continue;
        }
        produced = true;
        for chunk in pcm.chunks(rate.max(1)) {
            if !on_chunk(chunk.to_vec())? {
                return Ok(());
            }
        }
    }
    if !produced {
        return Err(anyhow!("Phonon produced no audio"));
    }
    Ok(())
}
