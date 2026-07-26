use std::path::Path;

pub const STARDIST_VERSION: &str = "0.9.2";

pub fn format_warning(
    message: &str,
    _category: &str,
    filename: &str,
    lineno: usize,
    _line: &str,
) -> String {
    let name = Path::new(filename)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(filename);
    format!("{name} ({lineno}): {message}\n")
}

pub fn _py_deprecation(
    ver_python: (u8, u8),
    ver_stardist: Option<&str>,
    current_python: (u8, u8),
    current_stardist: &str,
) -> Option<String> {
    let version_matches = current_python == ver_python;
    let stardist_before = if let Some(ver_stardist) = ver_stardist {
        let current_parts = current_stardist
            .split(|c: char| !c.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .map(|part| part.parse::<u64>().unwrap_or(0))
            .collect::<Vec<_>>();
        let requested_parts = ver_stardist
            .split(|c: char| !c.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .map(|part| part.parse::<u64>().unwrap_or(0))
            .collect::<Vec<_>>();
        let n = current_parts.len().max(requested_parts.len());
        let mut before = false;
        for i in 0..n {
            let current = *current_parts.get(i).unwrap_or(&0);
            let requested = *requested_parts.get(i).unwrap_or(&0);
            if current < requested {
                before = true;
                break;
            } else if current > requested {
                break;
            }
        }
        before
    } else {
        true
    };

    if version_matches && stardist_before {
        Some(format!(
            "You are using Python {}.{}, which is deprecated and will no longer be supported in {}.\n-> Please upgrade to Python {}.{} or later.",
            ver_python.0,
            ver_python.1,
            ver_stardist
                .map(|version| format!("StarDist {version}"))
                .unwrap_or_else(|| "future versions of StarDist".to_string()),
            ver_python.0,
            ver_python.1 + 1,
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_warning_matches_stardist_warning_style() {
        assert_eq!(
            format_warning("careful", "UserWarning", "/tmp/path/example.py", 17, ""),
            "example.py (17): careful\n"
        );
    }

    #[test]
    fn py_deprecation_warns_only_for_matching_python_and_old_stardist() {
        assert_eq!(
            _py_deprecation((3, 6), None, (3, 6), STARDIST_VERSION),
            Some(
                "You are using Python 3.6, which is deprecated and will no longer be supported in future versions of StarDist.\n-> Please upgrade to Python 3.7 or later."
                    .to_string()
            )
        );
        assert_eq!(
            _py_deprecation((3, 6), Some("1.0.0"), (3, 6), STARDIST_VERSION),
            Some(
                "You are using Python 3.6, which is deprecated and will no longer be supported in StarDist 1.0.0.\n-> Please upgrade to Python 3.7 or later."
                    .to_string()
            )
        );
        assert_eq!(
            _py_deprecation((3, 6), Some("0.9.2"), (3, 6), STARDIST_VERSION),
            None
        );
        assert_eq!(
            _py_deprecation((3, 6), None, (3, 7), STARDIST_VERSION),
            None
        );
    }
}
