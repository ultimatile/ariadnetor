use super::*;

#[test]
fn trunc_svd_error_and_target_err_arithmetic() {
    let bs = sample_known_svs(); // SVs = [3, 2, 1, 1]
    // chi_max=1 discards [2,1,1] → trunc_err = sqrt(4+1+1) = sqrt(6)
    let p1 = TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    };
    let (_, _, _, err) = trunc_svd_block_sparse(&backend(), &bs, 1, &p1).unwrap();
    assert!((err - 6.0f64.sqrt()).abs() < 1e-10, "trunc_err={err}");
    // target_err=0.5 → target_sq=0.25; smallest sv²=1 > 0.25 → all kept
    let p2 = TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(0.5),
    };
    let (_, sv, _, err2) = trunc_svd_block_sparse(&backend(), &bs, 1, &p2).unwrap();
    let kept: usize = sv.values.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(kept, 4);
    assert!(err2.abs() < 1e-12);
}
