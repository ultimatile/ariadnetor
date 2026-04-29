use super::super::{is_ascending_prefix, is_ascending_suffix, is_identity_perm};

// is_identity_perm
#[test]
fn identity_empty_is_true() {
    assert!(is_identity_perm(&[]));
}
#[test]
fn identity_natural_order_is_true() {
    assert!(is_identity_perm(&[0, 1, 2]));
}
#[test]
fn identity_swapped_is_false() {
    assert!(!is_identity_perm(&[1, 0]));
}
#[test]
fn identity_with_gap_is_false() {
    assert!(!is_identity_perm(&[0, 2]));
}

// is_ascending_prefix
#[test]
fn prefix_empty_is_true() {
    assert!(is_ascending_prefix(&[]));
}
#[test]
fn prefix_singleton_zero_is_true() {
    assert!(is_ascending_prefix(&[0]));
}
#[test]
fn prefix_singleton_nonzero_is_false() {
    assert!(!is_ascending_prefix(&[1]));
}
#[test]
fn prefix_natural_pair_is_true() {
    assert!(is_ascending_prefix(&[0, 1]));
}
#[test]
fn prefix_swapped_pair_is_false() {
    assert!(!is_ascending_prefix(&[1, 0]));
}
#[test]
fn prefix_with_gap_is_false() {
    assert!(!is_ascending_prefix(&[0, 2]));
}

// is_ascending_suffix
#[test]
fn suffix_empty_is_true() {
    assert!(is_ascending_suffix(&[], 3));
}
#[test]
fn suffix_last_singleton_is_true() {
    // rank=3, axes=[2]: offset=2, check 2==2.
    assert!(is_ascending_suffix(&[2], 3));
}
#[test]
fn suffix_non_last_singleton_is_false() {
    // rank=3, axes=[1]: offset=2, check 1==2 fails.
    assert!(!is_ascending_suffix(&[1], 3));
}
#[test]
fn suffix_last_two_is_true() {
    // rank=3, axes=[1, 2]: offset=1, check (1==1, 2==2).
    assert!(is_ascending_suffix(&[1, 2], 3));
}
#[test]
fn suffix_swapped_last_two_is_false() {
    // rank=3, axes=[2, 1]: offset=1, check (2==1) fails.
    assert!(!is_ascending_suffix(&[2, 1], 3));
}
#[test]
fn suffix_three_with_distinct_offset_is_true() {
    // rank=5, axes=[2,3,4]: offset=2.
    // Distinguishes `-` -> `+` (offset=8 -> false) and `-` -> `/` (offset=1 -> 2==1 false).
    // Distinguishes `+` -> `-` (2-0=2 ok, 2-1=1 != 3 fails) and `+` -> `*` (2*0=0 != 2 fails).
    assert!(is_ascending_suffix(&[2, 3, 4], 5));
}
#[test]
fn suffix_singleton_with_axes_len_one_is_true() {
    // rank=4, axes=[3]: offset=3.
    // Independently distinguishes `-` -> `/` (4/1=4 -> 3==4 false).
    assert!(is_ascending_suffix(&[3], 4));
}
