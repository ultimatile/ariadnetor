use super::*;

/// Backend that implements only the required methods, leaving every
/// optional operation on its trait default.
struct GemmOnlyBackend;

impl ComputeBackend for GemmOnlyBackend {
    fn name(&self) -> &'static str {
        "gemm-only"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn preferred_order(&self) -> MemoryOrder {
        MemoryOrder::ColumnMajor
    }

    fn gemm<T: Scalar>(&self, _desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
        Ok(())
    }

    fn transpose<T: Scalar>(&self, _desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
        Ok(())
    }
}

#[track_caller]
fn assert_not_supported(result: Result<(), BackendError>, op: &str) {
    match result {
        Err(BackendError::NotSupported(msg)) => assert_eq!(msg, op),
        other => panic!("{op}: expected Err(NotSupported), got {other:?}"),
    }
}

const ORDER: MemoryOrder = MemoryOrder::ColumnMajor;
const POLICY: ExecPolicy = ExecPolicy::Sequential;

#[test]
fn omitted_optional_operations_report_not_supported() {
    let b = GemmOnlyBackend;
    let a = [1.0_f64];

    let (mut u, mut s, mut vt) = ([0.0_f64], [0.0_f64], [0.0_f64]);
    assert_not_supported(
        b.svd(SvdDescriptor {
            m: 1,
            n: 1,
            a: &a,
            u: &mut u,
            s: &mut s,
            vt: &mut vt,
            order: ORDER,
            policy: POLICY,
        }),
        "svd",
    );

    let (mut q, mut r) = ([0.0_f64], [0.0_f64]);
    assert_not_supported(
        b.qr(QrDescriptor {
            m: 1,
            n: 1,
            a: &a,
            q: &mut q,
            r: &mut r,
            order: ORDER,
            policy: POLICY,
        }),
        "qr",
    );

    let (mut l, mut q) = ([0.0_f64], [0.0_f64]);
    assert_not_supported(
        b.lq(LqDescriptor {
            m: 1,
            n: 1,
            a: &a,
            l: &mut l,
            q: &mut q,
            order: ORDER,
            policy: POLICY,
        }),
        "lq",
    );

    let (mut w, mut v) = ([0.0_f64], [0.0_f64]);
    assert_not_supported(
        b.eigh(EighDescriptor {
            n: 1,
            a: &a,
            w: &mut w,
            v: &mut v,
            order: ORDER,
            policy: POLICY,
        }),
        "eigh",
    );

    let (mut w, mut v) = ([0.0_f64], [0.0_f64]);
    assert_not_supported(
        b.tridiag_eigh(TridiagEighDescriptor {
            n: 1,
            d: &a,
            e: &[],
            w: &mut w,
            v: &mut v,
            order: ORDER,
            policy: POLICY,
        }),
        "tridiag_eigh",
    );

    let (mut w, mut v) = ([Complex::new(0.0_f64, 0.0)], [Complex::new(0.0_f64, 0.0)]);
    assert_not_supported(
        b.eig(EigDescriptor {
            n: 1,
            a: &a,
            w: &mut w,
            v: &mut v,
            order: ORDER,
            policy: POLICY,
        }),
        "eig",
    );

    let mut x = [0.0_f64];
    assert_not_supported(
        b.solve(SolveDescriptor {
            n: 1,
            nrhs: 1,
            a: &a,
            b: &a,
            x: &mut x,
            order: ORDER,
            policy: POLICY,
        }),
        "solve",
    );
}
