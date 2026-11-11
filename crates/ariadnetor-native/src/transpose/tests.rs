//! Placement-token tests for the strided-perm route.
//!
//! `exec_in` is driven with a 4-thread pool built here, so the budgets below
//! do not depend on the host's core count.

use super::*;

fn pool4() -> Pool {
    Pool::builder().threads(4).build().unwrap()
}

#[test]
fn sequential_is_serial() {
    let pool = pool4();
    assert_eq!(
        exec_in(Some(&pool), ExecPolicy::Sequential).budget().get(),
        1
    );
}

#[test]
fn parallel_off_a_rayon_worker_uses_the_pool() {
    let pool = pool4();
    assert!(!on_rayon_worker());
    assert_eq!(
        exec_in(Some(&pool), ExecPolicy::Parallel(0)).budget().get(),
        4
    );
    assert_eq!(
        exec_in(Some(&pool), ExecPolicy::Parallel(3)).budget().get(),
        3
    );
    assert_eq!(
        exec_in(Some(&pool), ExecPolicy::Parallel(64))
            .budget()
            .get(),
        4
    );
}

#[test]
fn parallel_on_a_global_rayon_worker_is_serial() {
    let pool = pool4();
    rayon::scope(|_| {
        assert!(on_rayon_worker());
        assert_eq!(
            exec_in(Some(&pool), ExecPolicy::Parallel(0)).budget().get(),
            1
        );
        assert_eq!(
            exec_in(Some(&pool), ExecPolicy::Parallel(3)).budget().get(),
            1
        );
    });
}

#[test]
fn parallel_on_a_custom_rayon_worker_is_serial() {
    let pool = pool4();
    let custom = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    custom.install(|| {
        assert!(on_rayon_worker());
        assert_eq!(
            exec_in(Some(&pool), ExecPolicy::Parallel(0)).budget().get(),
            1
        );
    });
}

#[test]
fn parallel_without_a_pool_is_serial() {
    assert_eq!(exec_in(None, ExecPolicy::Parallel(0)).budget().get(), 1);
}

#[test]
fn exec_for_follows_the_shared_pool() {
    let width = pool().expect("pool builds").width().get();
    assert_eq!(width, rayon::current_num_threads());
    assert_eq!(exec_for(ExecPolicy::Sequential).budget().get(), 1);
    assert_eq!(exec_for(ExecPolicy::Parallel(0)).budget().get(), width);
    rayon::scope(|_| {
        assert_eq!(exec_for(ExecPolicy::Parallel(0)).budget().get(), 1);
    });
}
