//! Shared algorithms-level fixtures for this crate's tests, examples, and
//! benchmarks: U(1) MPS / MPO chains (`fixtures`) and Dense builders and
//! dense references (`dense_fixtures`), and at the root `PanicCase` and
//! `panic_message` for tables of panic cases. The tensor-layer densify /
//! leg helpers live in `ariadnetor_tensor::test_fixtures`.
//!
//! The module is gated on `cfg(any(test, feature = "test-fixtures"))`:
//! this crate's own unit tests reach it as `crate::test_fixtures` under
//! `cfg(test)`, while its integration tests, benchmarks, and example
//! tests are separate crates and reach it as
//! `ariadnetor_algorithms::test_fixtures` through the `test-fixtures`
//! feature, which a self dev-dependency enables. Being `pub` and
//! reachable from the library root also keeps every item live under
//! `dead_code = "forbid"`, whichever of those crates uses it.

use std::panic::{UnwindSafe, catch_unwind};

pub mod dense_fixtures;
pub mod fixtures;

/// A call expected to panic, for tables of panic cases.
pub type PanicCase = Box<dyn FnOnce() + UnwindSafe>;

/// The message `f` panics with.
///
/// # Panics
///
/// Panics if `f` returns, or panics with a payload that is not a string.
pub fn panic_message(f: impl FnOnce() + UnwindSafe) -> String {
    let payload = catch_unwind(f).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .expect("a string panic payload")
}

#[cfg(test)]
mod tests {
    use super::panic_message;

    #[test]
    fn panic_message_reads_string_and_str_payloads() {
        assert_eq!(panic_message(|| panic!("{}", 1)), "1");
        assert_eq!(panic_message(|| panic!("literal")), "literal");
    }

    #[test]
    #[should_panic(expected = "expected a panic")]
    fn panic_message_rejects_a_call_that_returns() {
        panic_message(|| {});
    }

    #[test]
    #[should_panic(expected = "a string panic payload")]
    fn panic_message_rejects_a_payload_that_is_not_a_string() {
        panic_message(|| std::panic::panic_any(1_u8));
    }
}
