use super::*;
use crate::test_util::assert_invalid_argument;

/// Rank-2 U1, identity flux: sector 0 is the 1×1 block `[4]`, sector 1 the
/// 2×2 block `diag(2, 1)`, so the singular values are `[4]` and `[2, 1]`.
fn two_sector_spectrum() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        square_legs(vec![(U1Sector(0), 1), (U1Sector(1), 2)]),
        U1Sector(0),
        order(),
    );
    bs.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[4.0]);
    bs.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[2.0, 0.0, 0.0, 1.0]);
    bs
}

fn trunc(
    bs: &BlockSparseTensorData<f64, U1Sector>,
    params: &TruncSvdParams,
) -> Result<BlockSparseTruncSvdResultBsp<f64, U1Sector>, LinalgError> {
    trunc_svd_block_sparse_with_policy_dense(&backend(), bs, 1, params, ExecPolicy::Sequential)
}

/// The norm over both sectors is `sqrt(21)`, so `Relative(0.25)` is a budget
/// of about 1.146 and discards the 1. Measured against sector 1 alone
/// (`sqrt(5)`), the budget would be about 0.559 and keep it.
#[test]
fn relative_target_is_measured_against_all_sectors() {
    let (_, sv, _, err) = trunc(
        &two_sector_spectrum(),
        &target_only(TruncErrTarget::Relative(0.25)),
    )
    .unwrap();
    let kept: usize = sv.values.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(kept, 2);
    assert!((err - 1.0).abs() < 1e-12, "{err}");
}

#[test]
fn invalid_target_values_are_rejected() {
    let bs = two_sector_spectrum();
    for value in [f64::NAN, -0.1, f64::INFINITY] {
        for t in [
            TruncErrTarget::Absolute(value),
            TruncErrTarget::Relative(value),
        ] {
            assert_invalid_argument(trunc(&bs, &target_only(t)), "target_trunc_err");
        }
    }
}

/// Parameters are checked before `nrow`, as in the dense kernel.
#[test]
fn invalid_parameters_are_reported_before_nrow() {
    let params = target_only(TruncErrTarget::Relative(f64::NAN));
    let result = trunc_svd_block_sparse_with_policy_dense(
        &backend(),
        &two_sector_spectrum(),
        0,
        &params,
        ExecPolicy::Sequential,
    );
    assert_invalid_argument(result, "target_trunc_err");
}

/// A flux no block can carry leaves no singular values; invalid parameters
/// are still rejected rather than passing through the empty truncation.
#[test]
fn invalid_parameters_are_rejected_without_singular_values() {
    let empty = BlockSparseTensorData::<f64, U1Sector>::zeros(
        square_legs(vec![(U1Sector(0), 1)]),
        U1Sector(5),
        order(),
    );
    let params = TruncSvdParams {
        chi_max: Some(0),
        target_trunc_err: None,
    };
    assert_invalid_argument(trunc(&empty, &params), "chi_max");
}
