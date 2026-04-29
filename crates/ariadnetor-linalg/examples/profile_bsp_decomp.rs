use std::time::Instant;

use rand::SeedableRng;

use arnet_linalg::{
    TruncSvdParams, lq_block_sparse, qr_block_sparse, svd_block_sparse, trunc_svd_block_sparse,
};
use arnet_native::NativeBackend;
use arnet_tensor::{BlockSparse, Direction, QNIndex, U1Sector};

fn random_bsp_matrix(q: usize, d: usize) -> BlockSparse<f64, U1Sector> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let sectors: Vec<(U1Sector, usize)> = (0..q as i32).map(|i| (U1Sector(i), d)).collect();
    let row = QNIndex::new(sectors.clone(), Direction::Out);
    let col = QNIndex::new(sectors, Direction::In);
    BlockSparse::random(vec![row, col], U1Sector(0), &mut rng)
}

fn bench_loop<F: FnMut()>(label: &str, iters: u32, mut f: F) {
    // Warm up
    f();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters;
    eprintln!("{label:40} {per_iter:>12.3?}  ({iters} iters, total {elapsed:.3?})");
}

fn main() {
    let backend = NativeBackend::new();

    // Decomposition is the dominant BSp cost per criterion data:
    // svd_bsp/q8_d64 ~5.4ms, trunc_svd_bsp/q8_d64 ~5.4ms,
    // qr/lq_bsp/q8_d64 ~2.1ms.
    for &(q, d) in &[(4, 64), (8, 64)] {
        // Aim for ~100ms total per loop so samply gets enough samples.
        let iters = if q * d >= 256 { 20 } else { 50 };
        let a = random_bsp_matrix(q, d);
        let chi_max = (q * d) / 2;
        let params = TruncSvdParams {
            chi_max: Some(chi_max),
            target_trunc_err: None,
        };

        eprintln!("\n=== q={q}, d={d}, total_dim={} ===", q * d);
        eprintln!(
            "  blocks={}, stored_elems={}",
            a.num_blocks(),
            a.stored_len()
        );

        bench_loop("  svd_block_sparse", iters, || {
            let _ = svd_block_sparse(&backend, &a, 1).unwrap();
        });

        bench_loop(
            &format!("  trunc_svd_block_sparse(chi_max={chi_max})"),
            iters,
            || {
                let _ = trunc_svd_block_sparse(&backend, &a, 1, &params).unwrap();
            },
        );

        bench_loop("  qr_block_sparse", iters, || {
            let _ = qr_block_sparse(&backend, &a, 1).unwrap();
        });

        bench_loop("  lq_block_sparse", iters, || {
            let _ = lq_block_sparse(&backend, &a, 1).unwrap();
        });
    }
}
