// -- Terminal capability detection -------------------------------------

use crate::error::TerminalError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Protocol {
    Kitty { animation_capable: bool },
    Iterm2,
}

/// `Ok(None)` means no known graphics-protocol terminal (`"auto"` only --
/// an explicit `"kitty"`/`"iterm2"` choice always succeeds or is rejected
/// outright): the caller should fall back to [`open_with_system_viewer`]
/// rather than attempting a text-based approximation, which this crate
/// deliberately no longer has (see the module doc comment for why).
pub(crate) fn resolve_protocol(value: &str) -> Result<Option<Protocol>, TerminalError> {
    match value {
        "auto" => Ok(detect_protocol_auto(&env_lookup)),
        "kitty" => Ok(Some(Protocol::Kitty {
            animation_capable: kitty_animation_capable(&env_lookup),
        })),
        "iterm2" => Ok(Some(Protocol::Iterm2)),
        other => Err(TerminalError::InvalidProtocol {
            value: other.to_string(),
        }),
    }
}

fn env_lookup(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Kitty-graphics-protocol terminals, in the documented priority order:
/// Kitty itself, Ghostty, cmux (Ghostty-based), and WezTerm.
fn kitty_protocol_terminal(get: &dyn Fn(&str) -> Option<String>) -> bool {
    let set = |name: &str| get(name).map(|value| !value.is_empty()).unwrap_or(false);
    let eq = |name: &str, value: &str| get(name).as_deref() == Some(value);
    set("KITTY_WINDOW_ID")
        || eq("TERM", "xterm-kitty")
        || eq("TERM_PROGRAM", "ghostty")
        || set("GHOSTTY_RESOURCES_DIR")
        || set("CMUX_WORKSPACE_ID")
        || set("CMUX_SURFACE_ID")
        || eq("TERM_PROGRAM", "WezTerm")
}

/// Whether a detected Kitty-protocol terminal also supports the Kitty
/// animation extension. WezTerm and plain Kitty do; Ghostty and cmux
/// (Ghostty-based) do not, as of this research -- so a GIF headed for
/// either of those falls back to simulated animation instead.
fn kitty_animation_capable(get: &dyn Fn(&str) -> Option<String>) -> bool {
    let set = |name: &str| get(name).map(|value| !value.is_empty()).unwrap_or(false);
    let eq = |name: &str, value: &str| get(name).as_deref() == Some(value);
    eq("TERM_PROGRAM", "WezTerm") || eq("TERM", "xterm-kitty") || set("KITTY_WINDOW_ID")
}

fn detect_protocol_auto(get: &dyn Fn(&str) -> Option<String>) -> Option<Protocol> {
    if kitty_protocol_terminal(get) {
        Some(Protocol::Kitty {
            animation_capable: kitty_animation_capable(get),
        })
    } else if get("TERM_PROGRAM").as_deref() == Some("iTerm.app") {
        Some(Protocol::Iterm2)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {

    use crate::protocol::Protocol;
    use crate::protocol::detect_protocol_auto;
    use crate::protocol::resolve_protocol;

    use std::collections::HashMap;

    #[test]
    fn detect_protocol_prefers_kitty_signals_in_priority_order() {
        let lookup = |pairs: &'static [(&'static str, &'static str)]| {
            let map: HashMap<&str, &str> = pairs.iter().copied().collect();
            move |name: &str| map.get(name).map(|value| value.to_string())
        };

        assert_eq!(
            detect_protocol_auto(&lookup(&[("KITTY_WINDOW_ID", "1")])),
            Some(Protocol::Kitty {
                animation_capable: true
            })
        );
        assert_eq!(
            detect_protocol_auto(&lookup(&[("TERM_PROGRAM", "ghostty")])),
            Some(Protocol::Kitty {
                animation_capable: false
            })
        );
        assert_eq!(
            detect_protocol_auto(&lookup(&[("CMUX_WORKSPACE_ID", "abc")])),
            Some(Protocol::Kitty {
                animation_capable: false
            })
        );
        assert_eq!(
            detect_protocol_auto(&lookup(&[("TERM_PROGRAM", "WezTerm")])),
            Some(Protocol::Kitty {
                animation_capable: true
            })
        );
        assert_eq!(
            detect_protocol_auto(&lookup(&[("TERM_PROGRAM", "iTerm.app")])),
            Some(Protocol::Iterm2)
        );
        // No known graphics-protocol terminal (Apple's Terminal.app
        // included -- see the module doc comment): `None`, not a
        // text-based fallback protocol.
        assert_eq!(detect_protocol_auto(&lookup(&[])), None);
    }

    #[test]
    fn resolve_protocol_rejects_unknown_values() {
        let error = resolve_protocol("bogus").unwrap_err();
        assert_eq!(error.code(), "invalid_protocol");
        // "ansi" was a valid choice before the text-based fallback was
        // removed; it must now be rejected the same as any other unknown
        // value, not silently accepted.
        let error = resolve_protocol("ansi").unwrap_err();
        assert_eq!(error.code(), "invalid_protocol");
    }

    #[test]
    fn resolve_protocol_honors_explicit_choices() {
        assert_eq!(resolve_protocol("iterm2").unwrap(), Some(Protocol::Iterm2));
        assert!(matches!(
            resolve_protocol("kitty").unwrap(),
            Some(Protocol::Kitty { .. })
        ));
    }
}
