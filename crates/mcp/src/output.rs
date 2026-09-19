use std::path::PathBuf;

pub(crate) const MAX_OUTPUT_PATH_BYTES: usize = 4 * 1024;

pub(crate) fn output_path(
    arguments: &serde_json::Value,
    extension: &str,
) -> Result<PathBuf, String> {
    let path = PathBuf::from(
        arguments
            .get("output_path")
            .and_then(serde_json::Value::as_str)
            .ok_or("output_path is required")?,
    );
    validate_output_path(&path, extension)?;
    Ok(path)
}

pub(crate) fn validate_output_path(path: &std::path::Path, extension: &str) -> Result<(), String> {
    if path.as_os_str().len() > MAX_OUTPUT_PATH_BYTES {
        return Err("output path exceeds 4 KiB".into());
    }
    if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case(extension))
    {
        Ok(())
    } else {
        Err(format!("output_path must use a .{extension} extension"))
    }
}
