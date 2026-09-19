use crate::*;

pub(crate) fn validate_positive(
    value: f32,
    name: &'static str,
) -> Result<(), SceneValidationError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(SceneValidationError::InvalidPositiveValue(name));
    }
    Ok(())
}

pub(crate) fn validate_color(color: Color) -> Result<(), SceneValidationError> {
    if color
        .iter()
        .all(|value| validate_unit_interval(*value).is_ok())
    {
        Ok(())
    } else {
        Err(SceneValidationError::InvalidColor)
    }
}

pub(crate) fn validate_unit_interval(value: f32) -> Result<(), SceneValidationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(SceneValidationError::InvalidColor)
    }
}

/// Validates an unconstrained `[f32; 2]` offset (e.g. `translate`): both
/// components must be finite, but -- unlike opacity or color -- there is no
/// range restriction, consistent with how a `Rect`'s `x`/`y` aren't
/// range-restricted today either.
pub(crate) fn validate_finite_pair(value: [f32; 2]) -> Result<(), SceneValidationError> {
    if value.iter().all(|component| component.is_finite()) {
        Ok(())
    } else {
        Err(SceneValidationError::InvalidTranslate)
    }
}
