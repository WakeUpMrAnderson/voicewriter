//! Resolves the bundled eSpeak NG binary + data directory, and (when the
//! platform DLL can be loaded) calls it in-process so phonemizing a clause is
//! a function call instead of starting `espeak-ng` (~0.4-1 s per process on
//! Windows). The binary itself is the phonemizer fallback for Kokoro TTS
//! (`--ipa` mode — see `kokoro.rs`'s module doc comment for why), while the
//! in-process library also serves the Piper engine's Persian voice
//! (`fa_phonemes` — see `piper.rs`); the fallback TTS role this module used
//! to serve (synthesizing non-Latin-script speech directly via eSpeak NG's
//! own voice) was removed when Kokoro became the sole TTS engine.

use std::path::PathBuf;
use tauri::{path::BaseDirectory, AppHandle, Manager, Wry};

/// Bundled eSpeak NG binary and its data directory, resolved via Tauri's
/// resource system (see `tauri.conf.json`'s `bundle.resources` and
/// `src-tauri/vendor/espeak-ng/`). Falls back to a system install on PATH
/// (e.g. via winget/apt/brew) if the bundled copy can't be resolved — useful
/// in dev builds run outside `tauri dev`'s resource-copying step.
pub(crate) struct Located {
    pub(crate) binary: PathBuf,
    pub(crate) data_dir: Option<PathBuf>,
}

/// Resolve the bundled/vendored/system eSpeak NG binary and its data dir.
pub(crate) fn locate(app: &AppHandle<Wry>) -> Located {
    let exe_name = if cfg!(windows) { "espeak-ng.exe" } else { "espeak-ng" };

    // 1) A real packaged build: Tauri's resource dir (see `bundle.resources`).
    let bundled_bin = app.path().resolve(exe_name, BaseDirectory::Resource).ok();
    let bundled_data = app
        .path()
        .resolve("espeak-ng-data", BaseDirectory::Resource)
        .ok();
    if let Some(binary) = bundled_bin.filter(|p| p.is_file()) {
        return Located { binary, data_dir: bundled_data.filter(|p| p.is_dir()) };
    }

    // 2) `tauri dev`: resources aren't copied, so fall back to the vendored
    // copy in the source tree directly (see `src-tauri/vendor/espeak-ng/`).
    if let Some(vendor_dir) = vendor_dir() {
        let binary = vendor_dir.join(exe_name);
        let data_dir = vendor_dir.join("espeak-ng-data");
        if binary.is_file() {
            return Located { binary, data_dir: data_dir.is_dir().then_some(data_dir) };
        }
    }

    // 3) Whatever's on PATH (a manual system install).
    eprintln!("espeak: no bundled or vendored copy found, falling back to PATH");
    Located { binary: PathBuf::from(exe_name), data_dir: None }
}

/// `src-tauri/vendor/espeak-ng/`, resolved relative to this crate's manifest
/// dir at compile time — only meaningful for dev builds run from a source
/// checkout, never a packaged release (which uses Tauri's resource dir
/// instead, per `locate()`'s first branch).
fn vendor_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor").join("espeak-ng");
    dir.is_dir().then_some(dir)
}

/// eSpeak NG loaded in-process, so phonemizing a clause is a function call
/// instead of starting `espeak-ng` (~0.4-1 s per process on Windows).
/// `pub(crate)` alongside `open_lib`/`lib_fa_phonemes` for `piper.rs`'s
/// model smoke test.
pub(crate) struct Lib {
    _lib: libloading::Library,
    set_voice: unsafe extern "C" fn(*const std::ffi::c_char) -> std::ffi::c_int,
    to_phonemes: unsafe extern "C" fn(
        *mut *const std::ffi::c_void,
        std::ffi::c_int,
        std::ffi::c_int,
    ) -> *const std::ffi::c_char,
    /// `espeak_TextToPhonemesWithTerminator` (same signature as
    /// `espeak_TextToPhonemes` plus a final `int *terminator` out-param,
    /// added in espeak-ng 1.52). Resolved opportunistically — `None` on older
    /// builds like the currently vendored one — because failing to load it
    /// must not break the existing en-us phonemization path.
    to_phonemes_with_terminator: Option<
        unsafe extern "C" fn(
            *mut *const std::ffi::c_void,
            std::ffi::c_int,
            std::ffi::c_int,
            *mut std::ffi::c_int,
        ) -> *const std::ffi::c_char,
    >,
    voice: String,
}

// SAFETY: eSpeak NG is not thread-safe; every call goes through the Mutex.
unsafe impl Send for Lib {}

static LIB: std::sync::OnceLock<Option<parking_lot::Mutex<Lib>>> = std::sync::OnceLock::new();

fn load_lib(app: &AppHandle<Wry>) -> Option<parking_lot::Mutex<Lib>> {
    use std::ffi::CString;
    let located = locate(app);
    #[cfg(windows)]
    let lib_path = located.binary.parent()?.join("libespeak-ng.dll");
    #[cfg(not(windows))]
    let lib_path = PathBuf::from("libespeak-ng.so.1");
    // The directory *containing* espeak-ng-data; null = the library's default.
    let data_parent = located
        .data_dir
        .as_ref()
        .and_then(|d| d.parent())
        // eSpeak NG can't open data under Windows' `\\?\` verbatim prefix,
        // which Tauri's resource paths carry.
        .and_then(|p| CString::new(strip_verbatim(&p.to_string_lossy())).ok());

    let result = open_lib(&lib_path, data_parent.as_ref());
    match result {
        Ok(lib) => {
            eprintln!("espeak: loaded library {}", lib_path.display());
            Some(parking_lot::Mutex::new(lib))
        }
        Err(e) => {
            eprintln!("espeak: library unavailable ({}: {e}), using the CLI", lib_path.display());
            None
        }
    }
}

fn strip_verbatim(path: &str) -> String {
    path.strip_prefix(r"\\?\").unwrap_or(path).to_string()
}

/// Load the eSpeak NG library and initialize it against `data_parent` (the
/// directory *containing* espeak-ng-data, or `None` for the library's
/// default). `pub(crate)` so `piper.rs`'s model smoke test can drive the
/// same in-process path without an `AppHandle`.
pub(crate) fn open_lib(
    lib_path: &std::path::Path,
    data_parent: Option<&std::ffi::CString>,
) -> Result<Lib, String> {
    use std::ffi::{c_char, c_int};
    {
        // SAFETY: symbol signatures match espeak-ng's speak_lib.h.
        unsafe {
            let lib = libloading::Library::new(lib_path).map_err(|e| e.to_string())?;
            let init: libloading::Symbol<unsafe extern "C" fn(c_int, c_int, *const c_char, c_int) -> c_int> =
                lib.get(b"espeak_Initialize").map_err(|e| e.to_string())?;
            const AUDIO_OUTPUT_RETRIEVAL: c_int = 1;
            const INITIALIZE_DONT_EXIT: c_int = 0x8000;
            let path_ptr = data_parent.map_or(std::ptr::null(), |c| c.as_ptr());
            if init(AUDIO_OUTPUT_RETRIEVAL, 0, path_ptr, INITIALIZE_DONT_EXIT) < 0 {
                return Err("espeak_Initialize failed".into());
            }
            let set_voice = *lib.get(b"espeak_SetVoiceByName").map_err(|e| e.to_string())?;
            let to_phonemes = *lib.get(b"espeak_TextToPhonemes").map_err(|e| e.to_string())?;
            // Optional: only espeak-ng >= 1.52 exports the terminator variant.
            let to_phonemes_with_terminator = lib.get(b"espeak_TextToPhonemesWithTerminator").ok().map(|s| *s);
            Ok(Lib { _lib: lib, set_voice, to_phonemes, to_phonemes_with_terminator, voice: String::new() })
        }
    }
}

/// IPA phonemes for `text` via the in-process library, or `None` if the
/// library couldn't be loaded (callers fall back to the CLI).
pub(crate) fn ipa(app: &AppHandle<Wry>, text: &str, voice: &str) -> Option<String> {
    let lib = LIB.get_or_init(|| load_lib(app)).as_ref()?;
    lib_ipa(&mut lib.lock(), text, voice)
}

fn lib_ipa(lib: &mut Lib, text: &str, voice: &str) -> Option<String> {
    set_voice(lib, voice)?;
    raw_phonemes(lib, text)
}

/// Point the shared library at `voice`, cached — `ipa` (en-us, for Kokoro)
/// and `fa_phonemes` (fa, for Piper) share one library, so the voice is
/// re-set through `espeak_SetVoiceByName` whenever the requested voice
/// changes.
fn set_voice(lib: &mut Lib, voice: &str) -> Option<()> {
    use std::ffi::CString;
    if lib.voice == voice {
        return Some(());
    }
    let v = CString::new(voice).ok()?;
    // SAFETY: `v` outlives the call; eSpeak NG copies the voice name.
    if unsafe { (lib.set_voice)(v.as_ptr()) } != 0 {
        return None;
    }
    lib.voice = voice.to_string();
    Some(())
}

/// One `espeak_TextToPhonemes` pass over `text` (voice must already be set):
/// eSpeak NG advances `ptr` through internal clause breaks until the text is
/// consumed, so multi-clause input yields several IPA strings — joined with
/// a space here, the same way `kokoro.rs` joins the CLI's line-per-clause
/// output.
fn raw_phonemes(lib: &mut Lib, text: &str) -> Option<String> {
    use std::ffi::{c_void, CStr, CString};
    const CHARS_UTF8: i32 = 1;
    const PHONEMES_IPA: i32 = 0x02;
    let text = CString::new(text).ok()?;
    // SAFETY: pointers are valid for the duration of each call; eSpeak NG
    // advances `ptr` through `text` and sets it to null when done.
    unsafe {
        let mut parts = Vec::new();
        let mut ptr = text.as_ptr() as *const c_void;
        while !ptr.is_null() {
            let out = (lib.to_phonemes)(&mut ptr, CHARS_UTF8, PHONEMES_IPA);
            if out.is_null() {
                break;
            }
            let s = CStr::from_ptr(out).to_string_lossy().trim().to_string();
            if !s.is_empty() {
                parts.push(s);
            }
        }
        Some(parts.join(" "))
    }
}

/// Per-clause `(IPA, terminator)` pairs for `text`, phonemized with eSpeak
/// NG's `fa` voice — the phonemizer Piper's Persian voice was trained on
/// (see `piper.rs`). Each IPA string is NFD-normalized before returning, so
/// every codepoint is a key of Piper's `phoneme_id_map`.
///
/// The terminator encodes the clause-break punctuation eSpeak NG stripped
/// from the IPA string: its low 16 bits are the punctuation character
/// (0 = none) and bit 0x80000 marks a sentence end (espeak-ng's
/// `CLAUSE_TYPE_SENTENCE`, as in `CLAUSE_PERIOD = 0x80000 + '.'`).
/// `None` if the library couldn't be loaded.
pub(crate) fn fa_phonemes(app: &AppHandle<Wry>, text: &str) -> Option<Vec<(String, i32)>> {
    let lib = LIB.get_or_init(|| load_lib(app)).as_ref()?;
    lib_fa_phonemes(&mut lib.lock(), text)
}

/// The locked inner half of `fa_phonemes` — `pub(crate)` so `piper.rs`'s
/// model smoke test can phonemize through a directly-`open_lib`'d library
/// without an `AppHandle`.
pub(crate) fn lib_fa_phonemes(lib: &mut Lib, text: &str) -> Option<Vec<(String, i32)>> {
    use std::ffi::{c_int, c_void, CStr, CString};
    use unicode_normalization::UnicodeNormalization;
    const CHARS_UTF8: i32 = 1;
    const PHONEMES_IPA: i32 = 0x02;
    set_voice(lib, "fa")?;
    match lib.to_phonemes_with_terminator {
        Some(to_phonemes_with_terminator) => {
            let text = CString::new(text).ok()?;
            // SAFETY: pointers are valid for the duration of each call;
            // eSpeak NG advances `ptr` through `text` and sets it to null
            // when done, writing each clause's break info into `terminator`.
            let mut clauses = Vec::new();
            unsafe {
                let mut ptr = text.as_ptr() as *const c_void;
                while !ptr.is_null() {
                    let mut terminator: c_int = 0;
                    let out = to_phonemes_with_terminator(&mut ptr, CHARS_UTF8, PHONEMES_IPA, &mut terminator);
                    if out.is_null() {
                        break;
                    }
                    let ipa = CStr::from_ptr(out).to_string_lossy().trim().nfd().collect::<String>();
                    // Punctuation-only clauses are kept: their terminator
                    // still carries a mark Piper must re-append.
                    if !ipa.is_empty() || terminator & 0xFFFF != 0 {
                        clauses.push((ipa, terminator));
                    }
                }
            }
            Some(clauses)
        }
        None => {
            // The vendored espeak-ng build predates
            // `espeak_TextToPhonemesWithTerminator` (espeak-ng 1.52), so
            // split the text into clauses ourselves and emulate the
            // terminator values eSpeak NG would report — the id-stream
            // semantics `piper.rs` sees stay identical either way.
            let mut clauses = Vec::new();
            for (piece, punct) in split_clauses(text) {
                let ipa = raw_phonemes(lib, &piece)?;
                clauses.push((ipa.nfd().collect::<String>(), emulated_terminator(punct)));
            }
            Some(clauses)
        }
    }
}

/// The clause-break punctuation eSpeak NG splits clauses at, including the
/// Arabic-script comma/semicolon/question mark Persian text uses.
const CLAUSE_PUNCT: &[char] = &['.', '!', '?', ',', ';', ':', '،', '؛', '؟'];

/// Split `text` into `(clause, boundary punctuation)` pieces on eSpeak NG's
/// clause breaks; the final piece's punctuation is `None` (end of text).
/// Decimal/number separators ("3.14", "1,000") are not treated as breaks,
/// mirroring eSpeak NG's own handling.
fn split_clauses(text: &str) -> Vec<(String, Option<char>)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<(String, Option<char>)> = Vec::new();
    let mut current = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let is_number_sep = (c == '.' || c == ',')
            && chars[..i].last().is_some_and(|p| p.is_numeric())
            && chars.get(i + 1).is_some_and(|n| n.is_numeric());
        if CLAUSE_PUNCT.contains(&c) && !is_number_sep {
            if !current.trim().is_empty() {
                out.push((std::mem::take(&mut current).trim().to_string(), Some(c)));
            } else if let Some(last) = out.last_mut() {
                // A punctuation run ("?!") — keep only the strongest mark.
                last.1 = Some(c);
            }
        } else {
            current.push(c);
        }
    }
    if !current.trim().is_empty() {
        out.push((std::mem::take(&mut current).trim().to_string(), None));
    }
    out
}

/// The terminator value eSpeak NG would report for a clause broken by
/// `punct` (`None` = end of text, which eSpeak NG treats as a period).
/// Arabic-script marks map to their Latin equivalents — those are what
/// eSpeak NG reports, and what Piper's id map carries (the Arabic variants
/// aren't keys in it).
fn emulated_terminator(punct: Option<char>) -> i32 {
    const SENTENCE_END: i32 = 0x80000;
    let c = match punct {
        None => return SENTENCE_END | ('.' as i32),
        Some('،') => ',',
        Some('؛') => ';',
        Some('؟') => '?',
        Some(c) => c,
    };
    match c {
        '.' | '!' | '?' => SENTENCE_END | (c as i32),
        _ => c as i32,
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    /// The in-process library must phonemize exactly like the CLI did.
    #[test]
    fn library_matches_cli() {
        let Some(dir) = vendor_dir() else { return };
        // Tauri resolves resource paths with the `\\?\` prefix; make sure that works.
        let data_parent =
            std::ffi::CString::new(strip_verbatim(&format!(r"\\?\{}", dir.display()))).unwrap();
        let mut lib = open_lib(&dir.join("libespeak-ng.dll"), Some(&data_parent)).unwrap();
        for text in ["Hello there", "The quick brown fox jumps over the lazy dog", "I read 42 books in 2024"] {
            let from_lib = lib_ipa(&mut lib, text, "en-us").unwrap();
            let out = std::process::Command::new(dir.join("espeak-ng.exe"))
                .current_dir(&dir)
                .args(["--path", ".", "-v", "en-us", "-q", "--ipa", text])
                .output()
                .unwrap();
            let from_cli = String::from_utf8_lossy(&out.stdout).lines().collect::<Vec<_>>().join(" ");
            assert_eq!(from_lib, from_cli.trim(), "[{text}]");
        }
    }
}

#[cfg(test)]
mod clause_tests {
    use super::*;

    #[test]
    fn split_clauses_splits_on_clause_punctuation() {
        // Each clause carries the punctuation that closed it; only a final
        // piece with no punctuation at all gets `None`.
        let clauses = split_clauses("سلام، حالت خوب؟");
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0], ("سلام".to_string(), Some('،')));
        assert_eq!(clauses[1], ("حالت خوب".to_string(), Some('؟')));
        // No punctuation at all: one clause, end of text.
        assert_eq!(split_clauses("سلام"), vec![("سلام".to_string(), None)]);
        assert_eq!(split_clauses("سلام. حالت خوب"), vec![
            ("سلام".to_string(), Some('.')),
            ("حالت خوب".to_string(), None)
        ]);
    }

    #[test]
    fn split_clauses_collapses_punctuation_runs_and_skips_empty() {
        // "؟!" is a run: only the strongest mark survives, no empty clause.
        let clauses = split_clauses("سلام؟!");
        assert_eq!(clauses, vec![("سلام".to_string(), Some('!'))]);
        // Leading punctuation with no preceding clause is dropped.
        assert_eq!(split_clauses("،سلام"), vec![("سلام".to_string(), None)]);
    }

    #[test]
    fn split_clauses_keeps_number_separators() {
        assert_eq!(split_clauses("3.14 و 1,000"), vec![("3.14 و 1,000".to_string(), None)]);
        // A sentence-ending period after a number still splits.
        let clauses = split_clauses("42. سال");
        assert_eq!(clauses, vec![("42".to_string(), Some('.')), ("سال".to_string(), None)]);
    }

    #[test]
    fn emulated_terminator_matches_espeak_clause_codes() {
        // espeak-ng: CLAUSE_PERIOD = CLAUSE_TYPE_SENTENCE + '.', clause marks
        // carry no sentence bit.
        const SENTENCE_END: i32 = 0x80000;
        assert_eq!(emulated_terminator(Some('.')), SENTENCE_END | ('.' as i32));
        assert_eq!(emulated_terminator(Some('!')), SENTENCE_END | ('!' as i32));
        assert_eq!(emulated_terminator(Some(',')), ',' as i32);
        assert_eq!(emulated_terminator(Some(':')), ':' as i32);
        assert_eq!(emulated_terminator(None), SENTENCE_END | ('.' as i32));
        // Arabic-script marks map to their Latin equivalents (which are what
        // Piper's phoneme_id_map carries).
        assert_eq!(emulated_terminator(Some('؟')), SENTENCE_END | ('?' as i32));
        assert_eq!(emulated_terminator(Some('،')), ',' as i32);
        assert_eq!(emulated_terminator(Some('؛')), ';' as i32);
    }
}
