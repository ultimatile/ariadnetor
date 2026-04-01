//! Sector trait and concrete types for abelian symmetries.
//!
//! A sector labels an irreducible representation of an abelian symmetry group.
//! The [`Sector`] trait provides fusion (tensor-product selection rule),
//! identity, and dual operations that drive block-sparse tensor construction.

use std::fmt::Debug;
use std::hash::Hash;

/// Abelian symmetry sector.
///
/// For abelian groups the fusion of two sectors is unique and commutative.
/// The following algebraic laws must hold for every implementation:
///
/// - **Identity**: `s.fuse(&S::identity()) == s`
/// - **Inverse**: `s.fuse(&s.dual()) == S::identity()`
/// - **Commutativity**: `s.fuse(&t) == t.fuse(&s)`
///
/// `Ord` is required so that [`QNIndex`](super::QNIndex) can keep sectors
/// in sorted order for binary search and merge operations.
pub trait Sector: Clone + Eq + Ord + Hash + Debug {
    /// Fuse two sectors (tensor-product selection rule).
    fn fuse(&self, other: &Self) -> Self;

    /// The identity (trivial) sector.
    fn identity() -> Self;

    /// The dual (conjugate) sector, mapping incoming ↔ outgoing.
    fn dual(&self) -> Self;
}

// ---------------------------------------------------------------------------
// Z2
// ---------------------------------------------------------------------------

/// Z₂ symmetry sector (values 0 or 1).
///
/// Fusion rule: (a + b) mod 2.
/// Every element is self-dual.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Z2Sector(u8);

impl Z2Sector {
    /// Create a Z₂ sector. Panics if `value` is not 0 or 1.
    pub fn new(value: u8) -> Self {
        assert!(value <= 1, "Z2Sector value must be 0 or 1, got {value}");
        Self(value)
    }

    /// Return the inner value (0 or 1).
    pub fn value(self) -> u8 {
        self.0
    }
}

impl Sector for Z2Sector {
    fn fuse(&self, other: &Self) -> Self {
        Self(self.0 ^ other.0)
    }

    fn identity() -> Self {
        Self(0)
    }

    fn dual(&self) -> Self {
        *self
    }
}

// ---------------------------------------------------------------------------
// U1
// ---------------------------------------------------------------------------

/// U(1) symmetry sector (integer charge).
///
/// Fusion rule: a + b.
/// Dual: −a.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U1Sector(pub i32);

impl Sector for U1Sector {
    fn fuse(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }

    fn identity() -> Self {
        Self(0)
    }

    fn dual(&self) -> Self {
        Self(-self.0)
    }
}

// ---------------------------------------------------------------------------
// Direct-product symmetry via tuples
// ---------------------------------------------------------------------------

impl<A: Sector, B: Sector> Sector for (A, B) {
    fn fuse(&self, other: &Self) -> Self {
        (self.0.fuse(&other.0), self.1.fuse(&other.1))
    }

    fn identity() -> Self {
        (A::identity(), B::identity())
    }

    fn dual(&self) -> Self {
        (self.0.dual(), self.1.dual())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the three algebraic laws for any Sector implementation.
    fn assert_sector_laws<S: Sector>(a: &S, b: &S) {
        let id = S::identity();

        // Identity
        assert_eq!(a.fuse(&id), *a);
        assert_eq!(id.fuse(a), *a);

        // Inverse
        assert_eq!(a.fuse(&a.dual()), id);
        assert_eq!(a.dual().fuse(a), id);

        // Commutativity
        assert_eq!(a.fuse(b), b.fuse(a));
    }

    #[test]
    fn z2_laws() {
        let s0 = Z2Sector::new(0);
        let s1 = Z2Sector::new(1);
        assert_sector_laws(&s0, &s1);
        assert_sector_laws(&s1, &s1);
    }

    #[test]
    fn z2_fusion_table() {
        let z0 = Z2Sector::new(0);
        let z1 = Z2Sector::new(1);
        assert_eq!(z0.fuse(&z0), z0);
        assert_eq!(z0.fuse(&z1), z1);
        assert_eq!(z1.fuse(&z0), z1);
        assert_eq!(z1.fuse(&z1), z0);
    }

    #[test]
    fn z2_dual() {
        assert_eq!(Z2Sector::new(0).dual(), Z2Sector::new(0));
        assert_eq!(Z2Sector::new(1).dual(), Z2Sector::new(1));
    }

    #[test]
    fn z2_ord() {
        assert!(Z2Sector::new(0) < Z2Sector::new(1));
    }

    #[test]
    #[should_panic(expected = "Z2Sector value must be 0 or 1")]
    fn z2_invalid_value() {
        Z2Sector::new(2);
    }

    #[test]
    fn u1_laws() {
        let s0 = U1Sector(0);
        let s1 = U1Sector(1);
        let s_neg = U1Sector(-3);
        assert_sector_laws(&s0, &s1);
        assert_sector_laws(&s1, &s_neg);
        assert_sector_laws(&s_neg, &s0);
    }

    #[test]
    fn u1_fusion() {
        assert_eq!(U1Sector(2).fuse(&U1Sector(3)), U1Sector(5));
        assert_eq!(U1Sector(-1).fuse(&U1Sector(1)), U1Sector(0));
    }

    #[test]
    fn u1_dual() {
        assert_eq!(U1Sector(3).dual(), U1Sector(-3));
        assert_eq!(U1Sector(0).dual(), U1Sector(0));
    }

    #[test]
    fn u1_ord() {
        assert!(U1Sector(-1) < U1Sector(0));
        assert!(U1Sector(0) < U1Sector(1));
    }

    #[test]
    fn tuple_laws() {
        let a = (U1Sector(1), Z2Sector::new(0));
        let b = (U1Sector(-2), Z2Sector::new(1));
        assert_sector_laws(&a, &b);
    }

    #[test]
    fn tuple_fusion() {
        let a = (U1Sector(1), Z2Sector::new(1));
        let b = (U1Sector(2), Z2Sector::new(1));
        assert_eq!(a.fuse(&b), (U1Sector(3), Z2Sector::new(0)));
    }

    #[test]
    fn tuple_identity_and_dual() {
        let id = <(U1Sector, Z2Sector)>::identity();
        assert_eq!(id, (U1Sector(0), Z2Sector::new(0)));

        let s = (U1Sector(3), Z2Sector::new(1));
        assert_eq!(s.dual(), (U1Sector(-3), Z2Sector::new(1)));
    }

    #[test]
    fn tuple_ord() {
        // Lexicographic: U1 compared first, then Z2
        let a = (U1Sector(0), Z2Sector::new(1));
        let b = (U1Sector(1), Z2Sector::new(0));
        assert!(a < b);
    }

    #[test]
    fn nested_tuple() {
        // (U1 × Z2) × U1 — verifies blanket impl composes
        let a = ((U1Sector(1), Z2Sector::new(0)), U1Sector(2));
        let b = ((U1Sector(-1), Z2Sector::new(1)), U1Sector(3));
        assert_sector_laws(&a, &b);
    }
}
