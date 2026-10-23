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

/// `HH:MM:SS.mmm` in local-independent UTC, so log files from different
/// machines can be compared without timezone math.
fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let total_seconds = now.as_secs();
    let millis = now.subsec_millis();

    let seconds = total_seconds % 60;
    let minutes = (total_seconds / 60) % 60;
    let hours = (total_seconds / 3600) % 24;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
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

/// Shortens a duration to seconds or minutes.
pub fn format_duration(duration: std::time::Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else {
        let minutes = (seconds / 60.0).floor();
        let rest = seconds - minutes * 60.0;
        format!("{:.0}m {:02.0}s", minutes, rest)
    }
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
    fn durations_switch_to_minutes() {
        assert_eq!(format_duration(Duration::from_millis(1500)), "1.5s");
        assert_eq!(format_duration(Duration::from_secs(75)), "1m 15s");
    }

    #[test]
    fn timestamps_are_fixed_width() {
        let stamp = timestamp();
        assert_eq!(stamp.len(), 12, "unexpected timestamp {stamp}");
    }
}
