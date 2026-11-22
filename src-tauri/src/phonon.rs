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

/// Synthesize `text` with Phonon, handing ~1 s slices of 24 kHz mono to
/// `on_chunk` (same contract as `kokoro::speak_chunks`; `Ok(false)` stops).
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
    let mut guard = cell.lock().map_err(|_| anyhow!("Phonon lock poisoned"))?;
    let tts = guard
        .as_mut()
        .ok_or_else(|| anyhow!("Phonon engine not initialized"))?;
    let pcm = tts.say(text).map_err(|e| anyhow!("Phonon synthesis failed: {e}"))?;
    if pcm.is_empty() {
        return Err(anyhow!("Phonon produced no audio"));
    }
    let rate = tts.sample_rate() as usize;
    let slice = rate; // ~1 s per slice — early playback, few seams
    for chunk in pcm.chunks(slice) {
        if !on_chunk(chunk.to_vec())? {
            return Ok(());
        }
    }
    Ok(())
}
