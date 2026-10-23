use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::ValueEnum;
use miette::{Context, IntoDiagnostic, Result};

/// How much chat-style progress to print.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum LogLevel {
    /// Nothing beyond the final summary.
    #[default]
    Off,
    /// One block per skill and one line per step.
    Brief,
    /// Every message, including raw model output and reasoning.
    Full,
}

impl LogLevel {
    pub fn is_enabled(self) -> bool {
        !matches!(self, LogLevel::Off)
    }

    pub fn is_full(self) -> bool {
        matches!(self, LogLevel::Full)
    }
}

impl FromStr for LogLevel {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(LogLevel::Off),
            "brief" => Ok(LogLevel::Brief),
            "full" => Ok(LogLevel::Full),
            other => Err(format!("unknown log level '{other}'")),
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            LogLevel::Off => "off",
            LogLevel::Brief => "brief",
            LogLevel::Full => "full",
        };
        formatter.write_str(name)
    }
}

struct Sink {
    file: Option<File>,
    tee_terminal: bool,
}

static SINK: OnceLock<Mutex<Sink>> = OnceLock::new();

fn sink() -> &'static Mutex<Sink> {
    SINK.get_or_init(|| {
        Mutex::new(Sink {
            file: None,
            tee_terminal: true,
        })
    })
}

/// Routes diagnostics to the terminal, a file, or both.
///
/// `None` leaves the terminal sink in place so nothing changes when the flag is
/// not used.
pub fn init(log_file: Option<&Path>, tee_terminal: bool) -> Result<()> {
    let file = match log_file {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .into_diagnostic()
                    .with_context(|| {
                        format!("Failed to create log directory {}", parent.display())
                    })?;
            }

            Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .into_diagnostic()
                    .with_context(|| format!("Failed to open log file {}", path.display()))?,
            )
        }
        None => None,
    };

    let mut current = sink()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    current.file = file;
    current.tee_terminal = tee_terminal;

    Ok(())
}

/// Writes one line to every configured destination.
///
/// A failure to write is deliberately ignored: logging must never abort an
/// audit that is otherwise making progress.
pub fn write_line(prefixed: &str, message: &str) {
    let mut guard = sink()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if let Some(file) = guard.file.as_mut() {
        for line in message.lines() {
            let _ = writeln!(file, "{} {}", timestamp(), line);
        }
        let _ = file.flush();
    }

    if guard.tee_terminal {
        let mut stderr = io::stderr().lock();
        let _ = writeln!(stderr, "{}{}", prefixed, message);
    }
}

/// RFC 3339 timestamp in UTC with an explicit `Z`, so a log that spans midnight
/// stays unambiguous and files from different machines compare directly.
fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = i64::try_from(now.as_secs()).unwrap_or(i64::MAX);
    let millis = now.subsec_millis();

    let datetime = chrono::DateTime::from_timestamp(seconds, 0)
        .unwrap_or_else(|| chrono::DateTime::from_timestamp(0, 0).expect("epoch is valid"));

    format!("{}.{:03}Z", datetime.format("%Y-%m-%dT%H:%M:%S"), millis)
}

/// Shortens a byte count to a human-readable size.
pub fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// The single duration format used by both the report and the logs.
pub fn format_duration_ms(milliseconds: u64) -> String {
    if milliseconds < 1_000 {
        return format!("{}ms", milliseconds);
    }

    let seconds = milliseconds as f64 / 1_000.0;
    if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else {
        let total_seconds = milliseconds / 1_000;
        format!("{}m {}s", total_seconds / 60, total_seconds % 60)
    }
}

/// Convenience wrapper for callers that already hold a `Duration`.
pub fn format_duration(duration: std::time::Duration) -> String {
    format_duration_ms(u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn log_levels_gate_output() {
        assert!(!LogLevel::Off.is_enabled());
        assert!(LogLevel::Brief.is_enabled());
        assert!(LogLevel::Full.is_enabled());
        assert!(!LogLevel::Brief.is_full());
        assert!(LogLevel::Full.is_full());
    }

    #[test]
    fn sizes_are_human_readable() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.0 KB");
        assert_eq!(format_bytes(3 * 1024 * 1024), "3.0 MB");
    }

    #[test]
    fn durations_share_one_format() {
        assert_eq!(format_duration_ms(500), "500ms");
        assert_eq!(format_duration_ms(1_500), "1.5s");
        assert_eq!(format_duration_ms(65_000), "1m 5s");
        assert_eq!(format_duration(Duration::from_millis(1500)), "1.5s");
        assert_eq!(format_duration(Duration::from_secs(75)), "1m 15s");
    }

    #[test]
    fn timestamps_include_date_and_zone() {
        let stamp = timestamp();

        // 2026-10-23T16:03:41.123Z
        assert_eq!(stamp.len(), 24, "unexpected timestamp {stamp}");
        assert!(stamp.ends_with('Z'), "zone marker missing: {stamp}");
        assert_eq!(&stamp[4..5], "-");
        assert_eq!(&stamp[10..11], "T");
        assert_eq!(&stamp[19..20], ".");
    }
}
