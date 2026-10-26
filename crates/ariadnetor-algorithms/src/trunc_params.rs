//! Entry-point validation of [`TruncSvdParams`] shared by the drivers.

use ariadnetor_linalg::TruncSvdParams;

/// Reject a `chi_max` of `Some(0)` and a `target_trunc_err` whose value is
/// not finite or is negative. `Err` names the constraint that failed.
pub(crate) fn validate_trunc_params(trunc: &TruncSvdParams) -> Result<(), &'static str> {
    if trunc.chi_max == Some(0) {
        return Err("trunc.chi_max must be > 0 if Some");
    }
    if let Some(target) = trunc.target_trunc_err {
        let te = target.value();
        if !te.is_finite() {
            return Err("trunc.target_trunc_err must be finite");
        }
        if te < 0.0 {
            return Err("trunc.target_trunc_err must be non-negative");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ariadnetor_linalg::TruncErrTarget;

    use super::*;

    fn params(chi_max: Option<usize>, target_trunc_err: Option<TruncErrTarget>) -> TruncSvdParams {
        TruncSvdParams {
            chi_max,
            target_trunc_err,
        }
    }

    const VARIANTS: [fn(f64) -> TruncErrTarget; 2] =
        [TruncErrTarget::Absolute, TruncErrTarget::Relative];

    #[test]
    fn accepts_the_boundary_values() {
        assert_eq!(validate_trunc_params(&params(None, None)), Ok(()));
        assert_eq!(validate_trunc_params(&params(Some(1), None)), Ok(()));
        for variant in VARIANTS {
            for p in [
                params(None, Some(variant(0.0))),
                params(Some(3), Some(variant(0.5))),
            ] {
                assert_eq!(validate_trunc_params(&p), Ok(()), "{p:?}");
            }
        }
    }

    #[test]
    fn names_the_failed_constraint() {
        assert_eq!(
            validate_trunc_params(&params(Some(0), None)),
            Err("trunc.chi_max must be > 0 if Some")
        );
        for variant in VARIANTS {
            let cases = [
                (f64::NAN, "trunc.target_trunc_err must be finite"),
                (f64::INFINITY, "trunc.target_trunc_err must be finite"),
                (-0.1, "trunc.target_trunc_err must be non-negative"),
            ];
            for (value, detail) in cases {
                let p = params(None, Some(variant(value)));
                assert_eq!(validate_trunc_params(&p), Err(detail), "{p:?}");
            }
        }
    }
}
