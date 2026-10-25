//! Entry-point validation of [`TruncSvdParams`] shared by the drivers that
//! forward them to a truncated SVD.

use ariadnetor_linalg::TruncSvdParams;

/// Reject a `chi_max` of `Some(0)` and a `target_trunc_err` that is not
/// finite or is negative. `Err` names the constraint that failed.
pub(crate) fn validate_trunc_params(trunc: &TruncSvdParams) -> Result<(), &'static str> {
    if trunc.chi_max == Some(0) {
        return Err("trunc.chi_max must be > 0 if Some");
    }
    if let Some(te) = trunc.target_trunc_err {
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
    use super::*;

    fn params(chi_max: Option<usize>, target_trunc_err: Option<f64>) -> TruncSvdParams {
        TruncSvdParams {
            chi_max,
            target_trunc_err,
        }
    }

    #[test]
    fn accepts_the_boundary_values() {
        for p in [
            params(None, None),
            params(Some(1), None),
            params(None, Some(0.0)),
            params(Some(3), Some(0.5)),
        ] {
            assert_eq!(validate_trunc_params(&p), Ok(()), "{p:?}");
        }
    }

    #[test]
    fn names_the_failed_constraint() {
        let cases = [
            (params(Some(0), None), "trunc.chi_max must be > 0 if Some"),
            (
                params(None, Some(f64::NAN)),
                "trunc.target_trunc_err must be finite",
            ),
            (
                params(None, Some(f64::INFINITY)),
                "trunc.target_trunc_err must be finite",
            ),
            (
                params(None, Some(-0.1)),
                "trunc.target_trunc_err must be non-negative",
            ),
        ];
        for (p, detail) in cases {
            assert_eq!(validate_trunc_params(&p), Err(detail), "{p:?}");
        }
    }
}
