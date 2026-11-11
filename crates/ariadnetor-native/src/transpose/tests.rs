//! Placement-token tests for the strided-perm route.
//!
//! The `exec_in` tests that pass a pool pass a 4-thread one built here, so
//! their budgets do not depend on the host's core count;
//! `exec_for_follows_the_shared_pool` checks the shared pool against the
//! host's Rayon width.

use super::*;

fn pool4() -> Pool {
    Pool::builder().threads(4).build().unwrap()
}

fn budget(exec: Exec<'_>) -> usize {
    exec.budget().get()
}

#[test]
fn sequential_is_serial() {
    let pool = pool4();
    assert_eq!(budget(exec_in(Some(&pool), ExecPolicy::Sequential)), 1);
}

#[test]
fn parallel_off_a_rayon_worker_uses_the_pool() {
    let pool = pool4();
    assert!(!on_rayon_worker());
    assert_eq!(budget(exec_in(Some(&pool), ExecPolicy::Parallel(0))), 4);
    assert_eq!(budget(exec_in(Some(&pool), ExecPolicy::Parallel(3))), 3);
    assert_eq!(budget(exec_in(Some(&pool), ExecPolicy::Parallel(64))), 4);
}

#[test]
fn parallel_on_a_global_rayon_worker_is_serial() {
    let pool = pool4();
    rayon::scope(|_| {
        assert!(on_rayon_worker());
        assert_eq!(budget(exec_in(Some(&pool), ExecPolicy::Parallel(0))), 1);
        assert_eq!(budget(exec_in(Some(&pool), ExecPolicy::Parallel(3))), 1);
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
        assert_eq!(budget(exec_in(Some(&pool), ExecPolicy::Parallel(0))), 1);
    });
}

#[test]
fn parallel_without_a_pool_is_serial() {
    assert_eq!(budget(exec_in(None, ExecPolicy::Parallel(0))), 1);
}

#[test]
fn exec_for_follows_the_shared_pool() {
    let width = pool().expect("pool builds").width().get();
    assert_eq!(width, rayon::current_num_threads());
    assert_eq!(budget(exec_for(ExecPolicy::Sequential)), 1);
    assert_eq!(budget(exec_for(ExecPolicy::Parallel(0))), width);
    rayon::scope(|_| {
        assert_eq!(budget(exec_for(ExecPolicy::Parallel(0))), 1);
    });
}
