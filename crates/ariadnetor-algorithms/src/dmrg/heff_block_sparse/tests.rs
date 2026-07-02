//! In-crate white-box unit tests for the BlockSparse / U(1) `heff`
//! per-step primitives (`EffectiveHamiltonian2SiteBlockSparse`,
//! `dmrg_2site_step_block_sparse`). They construct the operator and
//! drive the per-step entry point directly to assert on per-step
//! outputs — matvec vs a Dense oracle, eigenvalue, canonical form,
//! flux propagation, and error variants — so they live next to the
//! crate-internal code they exercise.
//!
//! Fixtures use the XY hopping interaction
//! `H = J (S+_a S-_{a+1} + S-_a S+_{a+1})`. The chain builders
//! (`make_n2_*` / `make_n3_*`) come from the `algorithms-fixtures`
//! crate and return `ariadnetor_mps` types; the environments are built
//! in-test via `BraketEnvs::build`.

use algorithms_fixtures::fixtures;

mod errors;
mod matvec;
