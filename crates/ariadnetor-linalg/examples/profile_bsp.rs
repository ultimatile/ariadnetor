use std::time::Instant;

use rand::SeedableRng;

use arnet_linalg::{contract_block_sparse, svd_block_sparse};
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

    for &(q, d) in &[(4, 64), (8, 16), (8, 64)] {
        let iters = if q * d >= 256 { 20 } else { 100 };
        let a = random_bsp_matrix(q, d);
        let b = random_bsp_matrix(q, d);

        eprintln!("\n=== q={q}, d={d}, total_dim={} ===", q * d);
        eprintln!(
            "  blocks={}, stored_elems={}",
            a.num_blocks(),
            a.stored_len()
        );

        bench_loop("  contract_bsp [1],[0] (aligned)", iters, || {
            let _ = contract_block_sparse(&backend, &a, &b, &[1], &[0]).unwrap();
        });

        bench_loop("  contract_bsp [0],[1] (permuted)", iters, || {
            let _ = contract_block_sparse(&backend, &a, &b, &[0], &[1]).unwrap();
        });

        bench_loop("  svd_block_sparse", iters, || {
            let _ = svd_block_sparse(&backend, &a, 1).unwrap();
        });
    }
}
