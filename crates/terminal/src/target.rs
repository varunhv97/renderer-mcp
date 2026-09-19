// -- Terminal output target resolution ----------------------------------

use crate::error::TerminalError;
use is_terminal::IsTerminal;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

/// Cap on how many ancestor processes to walk before giving up on finding a
/// real controlling terminal -- avoids any pathological loop in a broken
/// process tree.
// Ancestor-tty discovery (this and the helpers below) only runs on unix, where
// `ps` exists; the code still compiles everywhere so its tests do too.
#[cfg_attr(not(unix), allow(dead_code))]
const MAX_ANCESTOR_DEPTH: u32 = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TerminalTarget {
    Stdout,
    Device(PathBuf),
}

#[derive(Debug)]
pub(crate) enum TerminalResolution {
    Target(TerminalTarget),
    /// Stdout isn't a terminal, but a real tty device was found by walking
    /// this process's ancestors (see `discover_ancestor_tty_device`).
    /// Deliberately *not* folded into `Target(TerminalTarget::Device(_))`:
    /// whenever this branch is reached at all, it's because this process
    /// has no controlling terminal of its own -- almost always meaning it's
    /// running as a subprocess of some other interactive program (a coding
    /// agent's own TUI, most commonly) that is itself actively driving that
    /// same discovered device. Writing raw escape sequences directly into a
    /// tty this process doesn't control is unsafe in exactly that
    /// situation: the two writers' bytes can interleave mid-escape-sequence
    /// and corrupt the terminal emulator's parser state in a way not even a
    /// full terminal reset run the same (tty-less) way can recover, since
    /// `reset` itself needs the same ioctl access this process lacks.
    /// Confirmed live: a raw ANSI half-block write from exactly this kind
    /// of nested, controlling-terminal-less process left a real Terminal.app
    /// window showing solid stuck background color; `reset` run via the
    /// same subprocess path failed with "Inappropriate ioctl for device"
    /// and the window had to be closed and reopened. `run` handles this
    /// case via `open_with_system_viewer` instead. An explicit `--tty`
    /// override still resolves to `Target(TerminalTarget::Device)` and
    /// keeps the old inline-write behavior, since a caller passing that
    /// flag is knowingly taking responsibility for it being safe.
    #[cfg_attr(not(unix), allow(dead_code))]
    DiscoveredDevice,
    NoTerminal,
}

pub(crate) fn resolve_terminal_target(explicit_tty: Option<&Path>) -> TerminalResolution {
    if let Some(path) = explicit_tty {
        return TerminalResolution::Target(TerminalTarget::Device(path.to_path_buf()));
    }
    if std::io::stdout().is_terminal() {
        return TerminalResolution::Target(TerminalTarget::Stdout);
    }
    #[cfg(unix)]
    {
        if discover_ancestor_tty_device().is_some() {
            return TerminalResolution::DiscoveredDevice;
        }
    }
    TerminalResolution::NoTerminal
}

/// Best-effort: launches the OS's own default file opener (a separate GUI
/// process -- e.g. Preview.app via `open` on macOS) as a substitute for
/// writing image-protocol bytes into a discovered-but-unowned ancestor tty.
/// See [`TerminalResolution::DiscoveredDevice`]'s doc comment for why this
/// exists. Returns `true` iff the opener process was spawned successfully
/// (its own success/failure at actually displaying the file isn't
/// observable from here, same as any other fire-and-forget GUI launch).
/// `false` on an unsupported OS (anything but macOS/Linux) or if the opener
/// command itself couldn't be spawned (e.g. `xdg-open` not installed).
pub(crate) fn open_with_system_viewer(path: &Path) -> bool {
    let Some(opener) = system_opener_command() else {
        return false;
    };
    ProcessCommand::new(opener).arg(path).spawn().is_ok()
}

/// Which OS-native "open this file with its default app" command
/// [`open_with_system_viewer`] should use, if any. Split out as its own
/// pure function so this platform mapping is unit-testable independent of
/// actually spawning a process (which pops a real GUI window and so isn't
/// something a test should do).
fn system_opener_command() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("open")
    } else if cfg!(target_os = "linux") {
        Some("xdg-open")
    } else {
        None
    }
}

pub(crate) fn open_sink(target: &TerminalTarget) -> Result<Box<dyn Write>, TerminalError> {
    match target {
        TerminalTarget::Stdout => Ok(Box::new(std::io::stdout())),
        TerminalTarget::Device(path) => {
            let file = fs::OpenOptions::new()
                .write(true)
                .open(path)
                .map_err(|source| TerminalError::TtyOpen {
                    path: path.clone(),
                    source,
                })?;
            Ok(Box::new(file))
        }
    }
}

/// True when a `ps` `tty` column value names a real terminal device rather
/// than "no controlling terminal" -- BSD `ps` (macOS) prints `??`, GNU `ps`
/// (Linux) prints `?`, and both may print nothing at all.
#[cfg_attr(not(unix), allow(dead_code))]
fn is_real_tty_name(tty: &str) -> bool {
    let trimmed = tty.trim();
    !(trimmed.is_empty() || trimmed == "?" || trimmed == "??" || trimmed == "-")
}

/// Parses one line of `ps -o ppid=,tty= -p <pid>` output. Defensive about
/// whitespace since BSD and GNU `ps` pad columns differently.
#[cfg_attr(not(unix), allow(dead_code))]
fn parse_ps_ppid_tty(output: &str) -> Option<(u32, String)> {
    let line = output.lines().find(|line| !line.trim().is_empty())?;
    let mut fields = line.split_whitespace();
    let ppid = fields.next()?.parse::<u32>().ok()?;
    let tty = fields.next()?.to_string();
    Some((ppid, tty))
}

/// Walks the process ancestry starting at `start_pid`, calling `lookup`
/// (expected to behave like `ps -o ppid=,tty= -p <pid>`) for each ancestor
/// until one has a real controlling terminal, `ppid` reaches 0 or itself,
/// `lookup`/parsing fails, or `max_depth` is exceeded. `lookup` is injected
/// so this is testable against canned `ps` output instead of a live process
/// tree.
#[cfg_attr(not(unix), allow(dead_code))]
fn walk_ancestors_for_tty<F>(start_pid: u32, max_depth: u32, mut lookup: F) -> Option<String>
where
    F: FnMut(u32) -> Option<String>,
{
    let mut pid = start_pid;
    let mut visited = std::collections::HashSet::new();
    for _ in 0..max_depth {
        if !visited.insert(pid) {
            return None;
        }
        let raw = lookup(pid)?;
        let (ppid, tty) = parse_ps_ppid_tty(&raw)?;
        if is_real_tty_name(&tty) {
            return Some(tty);
        }
        if ppid == 0 || ppid == pid {
            return None;
        }
        pid = ppid;
    }
    None
}

#[cfg(unix)]
fn discover_ancestor_tty_device() -> Option<PathBuf> {
    let tty_name = walk_ancestors_for_tty(std::process::id(), MAX_ANCESTOR_DEPTH, |pid| {
        let output = ProcessCommand::new("ps")
            .args(["-o", "ppid=,tty=", "-p", &pid.to_string()])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&output.stdout).into_owned())
    })?;
    Some(PathBuf::from(format!("/dev/{tty_name}")))
}

#[cfg(test)]
mod tests {

    use crate::target::TerminalResolution;
    use crate::target::TerminalTarget;
    use crate::target::is_real_tty_name;
    use crate::target::parse_ps_ppid_tty;
    use crate::target::resolve_terminal_target;
    use crate::target::system_opener_command;
    use crate::target::walk_ancestors_for_tty;

    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    #[test]
    fn parse_ps_ppid_tty_handles_macos_and_linux_formatting() {
        assert_eq!(
            parse_ps_ppid_tty("  501 ttys008\n"),
            Some((501, "ttys008".to_string()))
        );
        assert_eq!(
            parse_ps_ppid_tty("1234 pts/3\n"),
            Some((1234, "pts/3".to_string()))
        );
        assert_eq!(parse_ps_ppid_tty("   42 ?\n"), Some((42, "?".to_string())));
        assert_eq!(parse_ps_ppid_tty(""), None);
        assert_eq!(parse_ps_ppid_tty("not-a-pid tty\n"), None);
    }

    #[test]
    fn is_real_tty_name_rejects_placeholder_values() {
        assert!(!is_real_tty_name("?"));
        assert!(!is_real_tty_name("??"));
        assert!(!is_real_tty_name(""));
        assert!(!is_real_tty_name("   "));
        assert!(is_real_tty_name("ttys008"));
        assert!(is_real_tty_name("pts/3"));
    }

    #[test]
    fn walk_ancestors_for_tty_finds_the_first_ancestor_with_a_real_tty() {
        // pid 300 (tty "?") -> pid 200 (tty "??") -> pid 100 (tty "ttys003")
        let responses: HashMap<u32, &str> =
            [(300, "200 ?\n"), (200, "100 ??\n"), (100, "1 ttys003\n")]
                .into_iter()
                .collect();
        let tty = walk_ancestors_for_tty(300, 32, |pid| responses.get(&pid).map(|s| s.to_string()));
        assert_eq!(tty, Some("ttys003".to_string()));
    }

    #[test]
    fn walk_ancestors_for_tty_gives_up_at_pid_1_with_no_real_tty() {
        let responses: HashMap<u32, &str> = [(50, "1 ?\n"), (1, "0 ?\n")].into_iter().collect();
        let tty = walk_ancestors_for_tty(50, 32, |pid| responses.get(&pid).map(|s| s.to_string()));
        assert_eq!(tty, None);
    }

    #[test]
    fn walk_ancestors_for_tty_stops_after_max_depth() {
        // ppid always climbs by one and never yields a real tty, pid 0, or
        // a self-loop, so only the max_depth cap can stop this.
        let tty = walk_ancestors_for_tty(1000, 5, |pid| Some(format!("{} ?\n", pid + 1)));
        assert_eq!(tty, None);
    }

    #[test]
    fn resolve_terminal_target_honors_an_explicit_tty_override() {
        let target = resolve_terminal_target(Some(Path::new("/tmp/fake-tty")));
        match target {
            TerminalResolution::Target(TerminalTarget::Device(path)) => {
                assert_eq!(path, PathBuf::from("/tmp/fake-tty"));
            }
            other => panic!("expected an explicit device target, got {other:?}"),
        }
    }

    /// Locks in `open_with_system_viewer`'s platform mapping without
    /// actually spawning a process (which would pop a real GUI window).
    #[test]
    fn system_opener_command_resolves_per_platform() {
        let opener = system_opener_command();
        if cfg!(target_os = "macos") {
            assert_eq!(opener, Some("open"));
        } else if cfg!(target_os = "linux") {
            assert_eq!(opener, Some("xdg-open"));
        } else {
            assert_eq!(opener, None);
        }
    }
}
