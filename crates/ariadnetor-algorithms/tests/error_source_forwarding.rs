//! Source-forwarding / non-duplication contract for the crate's
//! `thiserror`-derived error enums.
//!
//! Each wrapping variant must expose its child through `source()`
//! (so the chain stays walkable) while keeping its own `Display` to
//! the current layer — it must NOT fold the child's message into its
//! own `Display`, which would surface the cause twice in a
//! `source()`-walking reporter. These tests pin both halves: the
//! child is reachable via `source()`, and the wrapper's `Display`
//! does not contain the child's message.

use std::error::Error;

use ariadnetor_algorithms::dmrg::{
    DmrgEnvError, DmrgError, DmrgHeffError, DmrgSweepError, SweepDirection,
};
use ariadnetor_linalg::LinalgError;

/// A `LinalgError` carrying a distinctive marker token, so a
/// "wrapper Display does not contain the child message" assertion
/// cannot pass by coincidence.
fn marker_linalg_error() -> LinalgError {
    LinalgError::InvalidArgument("probe-marker-token".to_string())
}

/// Assert the standard contract for one wrapping error value: its
/// `Display` matches `expected_self_layer` exactly (so no child text
/// leaked in), it does not contain the child's message, and its
/// `source()` is the child.
fn assert_forwards(err: &dyn Error, expected_self_layer: &str, child_display: &str) {
    assert_eq!(
        err.to_string(),
        expected_self_layer,
        "wrapper Display must stay self-layer"
    );
    assert!(
        !err.to_string().contains(child_display),
        "wrapper Display must not duplicate the child message: {:?}",
        err.to_string()
    );
    let source = err
        .source()
        .expect("wrapping variant must expose its child via source()");
    assert_eq!(
        source.to_string(),
        child_display,
        "source() must reach the child error unchanged"
    );
}

#[test]
fn dmrg_error_env_forwards_to_source() {
    let child = DmrgEnvError::EmptyChain;
    let child_display = child.to_string();
    let err = DmrgError::Env(child);
    assert_forwards(&err, "DMRG environment build failed", &child_display);
}

#[test]
fn dmrg_error_sweep_forwards_to_source() {
    let child = DmrgSweepError::TooFewSites { n_sites: 1 };
    let child_display = child.to_string();
    let err = DmrgError::Sweep(child);
    assert_forwards(&err, "DMRG sweep driver failed", &child_display);
}

#[test]
fn dmrg_env_error_contract_forwards_to_source() {
    let child = marker_linalg_error();
    let child_display = child.to_string();
    let err = DmrgEnvError::Contract(child);
    assert_forwards(
        &err,
        "contract failure during DMRG environment update",
        &child_display,
    );
}

#[test]
fn dmrg_heff_error_contract_forwards_to_source() {
    let child = marker_linalg_error();
    let child_display = child.to_string();
    let err = DmrgHeffError::Contract(child);
    assert_forwards(
        &err,
        "linalg failure during two-site DMRG step",
        &child_display,
    );
}

#[test]
fn dmrg_sweep_error_step_forwards_to_source() {
    let child = DmrgHeffError::InvalidSite {
        site: 0,
        n_sites: 0,
    };
    let child_display = child.to_string();
    let err = DmrgSweepError::Step {
        sweep: 0,
        direction: SweepDirection::LeftToRight,
        site: 0,
        source: child,
    };
    assert_forwards(
        &err,
        "2-site DMRG step failed at sweep 0, LeftToRight, site 0",
        &child_display,
    );
}

#[test]
fn dmrg_sweep_error_env_forwards_to_source() {
    let child = DmrgEnvError::EmptyChain;
    let child_display = child.to_string();
    let err = DmrgSweepError::Env {
        sweep: 1,
        direction: SweepDirection::RightToLeft,
        site: 2,
        source: child,
    };
    assert_forwards(
        &err,
        "DmrgEnvs advance failed at sweep 1, RightToLeft, site 2",
        &child_display,
    );
}

#[test]
fn dmrg_sweep_error_scale_forwards_to_source() {
    let child = marker_linalg_error();
    let child_display = child.to_string();
    let err = DmrgSweepError::Scale {
        sweep: 0,
        direction: SweepDirection::LeftToRight,
        site: 0,
        source: child,
    };
    assert_forwards(
        &err,
        "S-absorb (diagonal scale) failed during sweep 0, LeftToRight, site 0",
        &child_display,
    );
}
