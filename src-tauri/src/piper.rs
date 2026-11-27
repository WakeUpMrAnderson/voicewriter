//! The Piper TTS engine (rhasspy/piper ONNX voices) — Persian, via the
//! "Mana" voice (fa_IR-mana-medium, downloaded through `model.rs`'s
//! "piper-fa-mana" entry). This gives VoiceWriter Persian speech, which the
//! English/Latin-script-only Kokoro engine can't produce.
//!
//! # Pipeline
//! Piper v1 is a standard espeak-phoneme model: text is phonemized with
//! eSpeak NG's `fa` voice (see `espeak.rs::fa_phonemes`), every NFD
//! codepoint of the returned IPA is mapped through the model config's
//! `phoneme_id_map`, and the id sequence runs through ONNX Runtime (same
//! `ort` setup as `kokoro.rs`) to produce 22.05 kHz mono PCM.
//!
//! # Differences from Kokoro's pipeline
//! - The phoneme vocabulary is NOT hardcoded — it lives in the model's JSON
//!   sidecar (`piper-fa-mana.onnx.json`), which also carries the audio
//!   sample rate and the training-time inference scales.
//! - eSpeak NG strips punctuation from the IPA string but reports it through
//!   the terminator out-param of `espeak_TextToPhonemesWithTerminator`; the
//!   mark is re-appended to the phoneme stream (Piper's vocabulary includes
//!   literal punctuation — see the `.`/`,`/`?` entries in the id map), and
//!   sentence terminators split synthesis into separate chunks just like
//!   Kokoro's dispatcher splits on punctuation itself.
//! - The currently vendored espeak-ng DLL predates the terminator API
//!   (espeak-ng 1.52); `espeak.rs::fa_phonemes` falls back to splitting
//!   clauses itself with equivalent terminator values, so this engine works
//!   with either build.
//!
//! The engine stays resident across calls (built once behind a mutex, like
//! `phonon.rs`'s Synth cache), and playback must use the config's sample
//! rate (see `sample_rate()`, read by `speak.rs`).

use anyhow::{anyhow, Context, Result};
use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use serde::Deserialize;
use std::collections::HashMap;
use tauri::{AppHandle, Wry};

// The three "special" ids every standard piper-v1 config assigns (Mana's
// included, verified against the shipped .onnx.json).
/// `^` — beginning-of-utterance marker.
const BOS: i64 = 1;
/// `$` — end-of-utterance marker.
const EOS: i64 = 2;
/// `_` — pad between phonemes, and after BOS.
const PAD: i64 = 0;

/// Sentence-end bit in an espeak clause terminator (espeak-ng's
/// `CLAUSE_TYPE_SENTENCE`, as in `CLAUSE_PERIOD = 0x80000 + '.'`).
const TERMINATOR_SENTENCE: i32 = 0x80000;

/// The model's `piper-fa-mana.onnx.json` — only the fields inference needs;
/// serde ignores the rest (dataset info, speaker map, phoneme_map, ...).
#[derive(Deserialize)]
struct PiperConfig {
    audio: Audio,
    inference: Inference,
    /// Maps each single-codepoint phoneme (NFD-decomposed espeak IPA) to the
    /// id(s) the model was trained to see for it — most phonemes map to one
    /// id, but the map allows several (serde_json preserves the exact
    /// Unicode key strings).
    phoneme_id_map: HashMap<String, Vec<i64>>,
}

/// The `audio` block — Mana's is `{ "sample_rate": 22050 }`.
#[derive(Deserialize)]
struct Audio {
    sample_rate: u32,
}

/// The `inference` block — the training-time scales, passed to the model's
/// `scales` input on every run (Mana: 0.667 / 1.0 / 0.8).
#[derive(Deserialize)]
struct Inference {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
}

/// Owns the loaded ONNX session and parsed config so repeated speak-aloud
/// calls don't reload the ~64 MB model file each time.
struct PiperEngine {
    session: Mutex<Session>,
    config: PiperConfig,
}

impl PiperEngine {
    /// Load the Mana model and its config from the models dir (`model.rs`).
    fn load(app: &AppHandle<Wry>) -> Result<Self> {
        let (model_path, config_path) = crate::model::piper_paths(app)?;
        for f in [&model_path, &config_path] {
            if !f.is_file() {
                return Err(anyhow!(
                    "Piper model not downloaded — add it in Settings → Voice Models"
                ));
            }
        }
        #[cfg(target_os = "linux")]
        init_runtime(app)?;
        Self::load_from(&model_path, &config_path)
    }

    /// Build the engine from explicit paths (`load`'s core, split out so the
    /// smoke test below can load from a path without an `AppHandle`).
    fn load_from(model_path: &std::path::Path, config_path: &std::path::Path) -> Result<Self> {
        let config: PiperConfig = serde_json::from_str(
            &std::fs::read_to_string(config_path)
                .with_context(|| format!("read Piper config {}", config_path.display()))?,
        )
        .with_context(|| format!("parse Piper config {}", config_path.display()))?;
        let session = Session::builder()
            .context("create ONNX session builder")?
            .with_intra_threads(default_threads())
            .map_err(|e| anyhow!("set ONNX threads: {e}"))?
            .commit_from_file(model_path)
            .with_context(|| format!("load Piper model {}", model_path.display()))?;
        eprintln!(
            "piper: model inputs = {:?}",
            session.inputs().iter().map(|i| i.name()).collect::<Vec<_>>()
        );
        eprintln!(
            "piper: model outputs = {:?}",
            session.outputs().iter().map(|o| o.name()).collect::<Vec<_>>()
        );
        Ok(Self { session: Mutex::new(session), config })
    }

    /// The config's audio sample rate (22050 for Mana) — read at load rather
    /// than hardcoded, since Piper voices ship different rates.
    fn sample_rate(&self) -> u32 {
        self.config.audio.sample_rate
    }

    /// Run ONNX inference for one sentence's id sequence: `input` (i64
    /// `[1, N]`), `input_lengths` (i64 `[1]`) and `scales` (f32 `[3]`, a
    /// 1-D tensor — this graph has no `sid` speaker input), yielding the
    /// model's raw f32 samples.
    fn synthesize(&self, ids: &[i64]) -> Result<Vec<f32>> {
        let n = ids.len();
        let input = Array2::from_shape_vec((1, n), ids.to_vec()).context("build input tensor")?;
        let input_lengths = Array1::from_vec(vec![n as i64]);
        let scales = Array1::from_vec(vec![
            self.config.inference.noise_scale,
            self.config.inference.length_scale,
            self.config.inference.noise_w,
        ]);
        let mut session = self.session.lock();
        let input_value = Tensor::from_array(input).context("input tensor")?;
        let lengths_value = Tensor::from_array(input_lengths).context("input_lengths tensor")?;
        let scales_value = Tensor::from_array(scales).context("scales tensor")?;
        let outputs = session
            .run(ort::inputs![
                "input" => input_value,
                "input_lengths" => lengths_value,
                "scales" => scales_value,
            ])
            .map_err(|e| anyhow!("run Piper inference: {e}"))?;
        let audio = outputs
            .get("output")
            .ok_or_else(|| anyhow!("Piper model produced no 'output' tensor"))?;
        let (shape, data) = audio
            .try_extract_tensor::<f32>()
            .context("extract Piper audio tensor")?;
        Ok(squeeze(shape, data))
    }
}

/// The Piper ONNX export wraps the waveform in singleton dimensions (a
/// `[1, T, 1]`-style artifact rather than a plain `[T]`); the buffer is
/// contiguous, so the flat 1-D sample vector is exactly the first
/// `num_elements` samples of the extracted data.
fn squeeze(shape: &[i64], data: &[f32]) -> Vec<f32> {
    let mut n: usize = 1;
    for &d in shape {
        n = n.saturating_mul(d.max(0) as usize);
    }
    data[..n.min(data.len())].to_vec()
}

/// Peak-normalize to ±1 (Piper's raw output is much quieter than Kokoro's)
/// and clamp to [-1, 1]; a silent chunk (peak < 1e-8) is zeroed rather than
/// amplified into noise.
fn normalize(samples: &mut [f32]) {
    let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak < 1e-8 {
        samples.fill(0.0);
        return;
    }
    for s in samples.iter_mut() {
        *s = (*s / peak).clamp(-1.0, 1.0);
    }
}

/// Build Piper's training-faithful id sequence for one sentence's phonemes:
/// BOS + PAD first, then every mapped phoneme's id(s) each followed by a
/// PAD, then EOS. Codepoints missing from the map are skipped with a
/// warning — piper's own policy (never error); this is safe because every
/// mapped phoneme is PAD-delimited, so a skipped one leaves a clean gap.
fn phoneme_ids(phonemes: &[char], map: &HashMap<String, Vec<i64>>) -> Vec<i64> {
    let mut ids = vec![BOS, PAD];
    for p in phonemes {
        match map.get(&p.to_string()) {
            Some(mapped) => {
                ids.extend_from_slice(mapped);
                ids.push(PAD);
            }
            None => eprintln!("piper: phoneme '{p}' not in the id map, skipping"),
        }
    }
    ids.push(EOS);
    ids
}

/// Assemble espeak's per-clause `(IPA, terminator)` pairs into per-sentence
/// phoneme lists: clauses accumulate until a sentence-end terminator, with
/// each clause's punctuation character re-appended to the stream (eSpeak NG
/// strips it from the IPA) — and a word-space after the clause marks ',', ':'
/// and ';' so the next clause doesn't glue on. This mirrors piper's own
/// terminator mapping.
fn sentences_from_clauses(clauses: &[(String, i32)]) -> Vec<Vec<char>> {
    let mut sentences = Vec::new();
    let mut current: Vec<char> = Vec::new();
    for (ipa, terminator) in clauses {
        current.extend(ipa.chars());
        let punct = char::from_u32((terminator & 0xFFFF) as u32).filter(|&c| c != '\0');
        match punct {
            Some(',') | Some(':') | Some(';') => {
                current.push(punct.unwrap());
                current.push(' ');
            }
            Some(c) => current.push(c),
            None => {}
        }
        if terminator & TERMINATOR_SENTENCE != 0 && !current.is_empty() {
            sentences.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    sentences
}

/// Process-wide lazily-loaded engine, shared across calls so the model stays
/// resident in memory after first use.
static ENGINE: std::sync::OnceLock<parking_lot::Mutex<Option<std::sync::Arc<PiperEngine>>>> =
    std::sync::OnceLock::new();

fn engine(app: &AppHandle<Wry>) -> Result<std::sync::Arc<PiperEngine>> {
    let cell = ENGINE.get_or_init(|| parking_lot::Mutex::new(None));
    let mut guard = cell.lock();
    if let Some(e) = guard.as_ref() {
        return Ok(e.clone());
    }
    let e = std::sync::Arc::new(PiperEngine::load(app)?);
    *guard = Some(e.clone());
    Ok(e)
}

/// The loaded engine's config sample rate (22050 for Mana), so playback can
/// hand Piper's audio to rodio at its true rate (see `speak.rs`, which plays
/// Kokoro's and Phonon's 24 kHz output at `kokoro::SAMPLE_RATE`). `None`
/// while the engine isn't built — callers fall back to Kokoro's rate.
pub fn sample_rate() -> Option<u32> {
    let cell = ENGINE.get()?;
    cell.lock().as_ref().map(|e| e.sample_rate())
}

/// Load the model in the background when Piper is the selected engine
/// (dispatched from `kokoro::preload`, which the app calls at startup and on
/// engine switches): builds the engine and synthesizes a short Persian
/// warm-up so the first real speak pays neither the model load nor ONNX
/// Runtime's kernel warm-up. Silent on failure — the real error surfaces on
/// the next speak.
pub fn preload(app: AppHandle<Wry>) {
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        match speak_chunks(&app, "سلام.", "", |_| Ok(true)) {
            Ok(()) => eprintln!("piper: preloaded in {:?}", t0.elapsed()),
            Err(e) => eprintln!("piper: preload failed: {e}"),
        }
    });
}

/// Drop the loaded model from memory (e.g. after the user deletes it from
/// Settings → Voice Models) so a stale session isn't reused.
pub fn unload() {
    if let Some(cell) = ENGINE.get() {
        *cell.lock() = None;
    }
}

/// Synthesize `text` (Persian) sentence by sentence: phonemize into
/// sentences via eSpeak NG's `fa` voice, build each sentence's id sequence,
/// run ONNX, peak-normalize, and hand ~1 s (`sample_rate()` samples) slices
/// of each sentence to `on_chunk` as soon as it's ready so playback can
/// start after the first sentence (`Ok(false)` stops early). `voice` is
/// ignored — Mana is a single-voice model.
pub fn speak_chunks(
    app: &AppHandle<Wry>,
    text: &str,
    _voice: &str,
    mut on_chunk: impl FnMut(Vec<f32>) -> Result<bool>,
) -> Result<()> {
    if text.trim().is_empty() {
        return Err(anyhow!("nothing to speak"));
    }
    let eng = engine(app)?;
    let clauses = crate::espeak::fa_phonemes(app, text).ok_or_else(|| {
        anyhow!("espeak-ng library unavailable — Piper needs the bundled espeak-ng to phonemize Persian")
    })?;
    let mut produced = false;
    for phonemes in sentences_from_clauses(&clauses) {
        let ids = phoneme_ids(&phonemes, &eng.config.phoneme_id_map);
        if ids.len() <= 3 {
            continue; // just BOS/PAD/EOS — nothing speakable in this sentence
        }
        let t0 = std::time::Instant::now();
        let mut samples = eng.synthesize(&ids)?;
        normalize(&mut samples);
        eprintln!(
            "piper: sentence {} phonemes / {} ids: synth {:?}, audio {:.1}s",
            phonemes.len(),
            ids.len(),
            t0.elapsed(),
            samples.len() as f64 / eng.sample_rate() as f64
        );
        if samples.is_empty() {
            continue;
        }
        produced = true;
        for chunk in samples.chunks(eng.sample_rate() as usize) {
            if !on_chunk(chunk.to_vec())? {
                return Ok(());
            }
        }
    }
    if !produced {
        return Err(anyhow!("couldn't phonemize this text"));
    }
    Ok(())
}

/// ONNX Runtime threads: physical-ish core count. Using every logical core
/// (hyper-threads) slows inference down through contention (same rationale
/// as `kokoro.rs`).
fn default_threads() -> usize {
    let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
    (logical / 2).clamp(1, 8)
}

/// Linux loads ONNX Runtime at run time from the copy bundled with the app —
/// same reason as `kokoro.rs::init_runtime` (kept as a separate copy rather
/// than made `pub(crate)` so each engine owns its init; ort's
/// `EnvironmentBuilder::commit` is idempotent, so whichever engine runs
/// first wins).
#[cfg(target_os = "linux")]
fn init_runtime(app: &AppHandle<Wry>) -> Result<()> {
    use std::sync::OnceLock;
    use tauri::Manager;
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let lib = app
            .path()
            .resource_dir()
            .map_err(|e| e.to_string())?
            .join("libonnxruntime.so");
        let _ = ort::init_from(&lib)
            .map_err(|e| format!("load ONNX Runtime {}: {e}", lib.display()))?
            .commit();
        Ok(())
    })
    .clone()
    .map_err(|e| anyhow!(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny fixture standing in for the real 157-key `phoneme_id_map`
    /// (Mana's real values for these keys: '^'→[1], '_'→[0], '$'→[2],
    /// ' '→[3], 'a'→[14], 's'→[31]).
    fn fixture_map() -> HashMap<String, Vec<i64>> {
        [
            ("^", vec![1]),
            ("_", vec![0]),
            ("$", vec![2]),
            (" ", vec![3]),
            ("a", vec![14]),
            ("s", vec![31]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    #[test]
    fn ids_start_with_bos_pad_and_end_with_eos() {
        let ids = phoneme_ids(&['a', 's'], &fixture_map());
        // [1, 0] = BOS + PAD; each phoneme's id then a PAD; [2] = EOS.
        assert_eq!(ids, vec![1, 0, 14, 0, 31, 0, 2]);
    }

    #[test]
    fn ids_skip_unmapped_phonemes_with_warning() {
        // 'z' has no entry in the fixture map — piper skips it, never errors.
        let ids = phoneme_ids(&['a', 'z', 's'], &fixture_map());
        assert_eq!(ids, vec![1, 0, 14, 0, 31, 0, 2]);
        // A word space maps like any other phoneme (" "→[3]).
        let ids = phoneme_ids(&['s', ' ', 'a'], &fixture_map());
        assert_eq!(ids, vec![1, 0, 31, 0, 3, 0, 14, 0, 2]);
    }

    #[test]
    fn multi_codepoint_ipa_maps_codepoint_by_codepoint() {
        // espeak's raw IPA carries multi-codepoint clusters (e.g. "sˈɑ" is
        // 's' + stress-mark + vowel); each NFD codepoint maps separately.
        let ids = phoneme_ids(&['s', 'ˈ', 'ɒ'], &fixture_map());
        assert_eq!(ids, vec![1, 0, 31, 0, 2]); // ˈ/ɒ unmapped in the fixture → skipped
    }

    #[test]
    fn nfd_splits_composed_persian_characters() {
        use unicode_normalization::UnicodeNormalization;
        // U+0623 (alef with hamza above) canonically decomposes into alef +
        // combining hamza — exactly the per-codepoint phonemes the id map
        // expects. Plain ASCII and already-decomposed IPA pass through.
        let nfd: Vec<char> = "أ".nfd().collect();
        assert_eq!(nfd, vec!['ا', '\u{0654}']);
        assert_eq!("salˈɑm".nfd().collect::<String>(), "salˈɑm");
    }

    #[test]
    fn clauses_assemble_into_sentences_with_punctuation() {
        // "سلام، حالت خوب؟" as espeak would report it: clause 1 breaks on ','
        // (no sentence bit), clause 2 on '?' (sentence bit 0x80000).
        const SENT: i32 = 0x80000;
        let clauses = vec![
            ("salˈɑm".to_string(), ',' as i32),
            ("hˈɑlat xˈuːb".to_string(), SENT | ('?' as i32)),
        ];
        let sentences = sentences_from_clauses(&clauses);
        assert_eq!(sentences.len(), 1);
        // Punctuation re-appended, word-space after the ',' clause mark.
        let s: String = sentences[0].iter().collect();
        assert_eq!(s, "salˈɑm, hˈɑlat xˈuːb?");
    }

    #[test]
    fn clauses_split_on_sentence_terminators() {
        const SENT: i32 = 0x80000;
        let clauses = vec![
            ("ˈyek".to_string(), SENT | ('.' as i32)),
            ("dˈo".to_string(), SENT | ('.' as i32)),
        ];
        let sentences = sentences_from_clauses(&clauses);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].iter().collect::<String>(), "ˈyek.");
        assert_eq!(sentences[1].iter().collect::<String>(), "dˈo.");
        // No sentence bit at all: everything accumulates into one sentence.
        let clauses = vec![("a".to_string(), ':' as i32), ("b".to_string(), SENT | ('.' as i32))];
        let sentences = sentences_from_clauses(&clauses);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].iter().collect::<String>(), "a: b.");
    }

    #[test]
    fn clauses_without_punctuation_or_terminator_flush_at_the_end() {
        // terminator 0 = no punctuation mark; the trailing accumulator still
        // flushes as a final sentence.
        let clauses = vec![("s".to_string(), 0), ("l".to_string(), 0)];
        let sentences = sentences_from_clauses(&clauses);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].iter().collect::<String>(), "sl");
    }

    #[test]
    fn normalize_peak_scales_and_clamps() {
        // Quietest chunk becomes full-scale.
        let mut a = vec![0.5, -0.25, 0.0];
        normalize(&mut a);
        assert_eq!(a, vec![1.0, -0.5, 0.0]);
        // Values beyond ±1 (shouldn't happen, but clamped regardless).
        let mut b = vec![2.0, -4.0];
        normalize(&mut b);
        assert_eq!(b, vec![0.5, -1.0]);
        // Near-silence is zeroed, not amplified into noise.
        let mut c = vec![1e-9, -1e-10];
        normalize(&mut c);
        assert_eq!(c, vec![0.0, 0.0]);
    }

    #[test]
    fn squeeze_flattens_singleton_dimensions() {
        // A [1, 5, 1] export artifact holds exactly 5 contiguous samples.
        assert_eq!(squeeze(&[1, 5, 1], &[1.0, 2.0, 3.0, 4.0, 5.0]).len(), 5);
        assert_eq!(squeeze(&[1, 3], &[9.0, 8.0, 7.0]), vec![9.0, 8.0, 7.0]);
        // A zero dimension means no samples at all.
        assert!(squeeze(&[1, 0, 1], &[1.0]).is_empty());
    }

    /// End-to-end smoke test against the real Mana model and the vendored
    /// espeak-ng library: phonemize a Persian sentence, build ids, run ONNX
    /// inference, and check the audio comes back at full scale. Exercises
    /// everything `speak_chunks` does except Tauri's path resolution.
    /// `cargo test smoke_real_piper -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the downloaded Piper model + vendored espeak-ng"]
    fn smoke_real_piper() {
        // The espeak-ng side, wired the same way `espeak::load_lib` does it
        // (the directory containing espeak-ng-data as the data parent).
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor").join("espeak-ng");
        let data_parent = std::ffi::CString::new(dir.to_string_lossy().as_ref()).expect("data parent");
        let mut lib = crate::espeak::open_lib(&dir.join("libespeak-ng.dll"), Some(&data_parent))
            .expect("open espeak-ng library");
        let text = "سلام. این یک آزمایش است، بی‌صدا نیست.";
        let clauses = crate::espeak::lib_fa_phonemes(&mut lib, text).expect("fa phonemes");
        for (ipa, term) in &clauses {
            println!("clause {ipa:?} terminator {term:#x}");
        }

        // The Piper side, from the real downloaded model files.
        let models = std::path::PathBuf::from(std::env::var("APPDATA").unwrap())
            .join("com.voicewriter.app")
            .join("models");
        let eng = PiperEngine::load_from(
            &models.join("piper-fa-mana.onnx"),
            &models.join("piper-fa-mana.onnx.json"),
        )
        .unwrap();
        assert_eq!(eng.sample_rate(), 22_050);

        let mut produced = 0;
        for phonemes in sentences_from_clauses(&clauses) {
            let ids = phoneme_ids(&phonemes, &eng.config.phoneme_id_map);
            assert!(ids.len() > 3, "no mapped phonemes for {:?}", phonemes);
            let t0 = std::time::Instant::now();
            let mut samples = eng.synthesize(&ids).unwrap();
            normalize(&mut samples);
            println!(
                "{} phonemes / {} ids -> {:.1}s audio in {:?}, peak {:.3}",
                phonemes.len(),
                ids.len(),
                samples.len() as f64 / eng.sample_rate() as f64,
                t0.elapsed(),
                samples.iter().fold(0.0f32, |m, s| m.max(*s)),
            );
            assert!(!samples.is_empty());
            produced += 1;
        }
        assert_eq!(produced, 2, "expected two sentences (period + comma-clause end)");
    }
}
