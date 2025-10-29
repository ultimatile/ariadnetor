//! Sector types for block-sparse tensors with symmetries

/// Z2 symmetry sector (0 or 1)
///
/// User defines meaning (even/odd, up/down, etc.)
/// Fusion rule: (a + b) mod 2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Z2Sector(pub u8);

/// U(1) symmetry sector (integer charge)
///
/// Fusion rule: a + b
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct U1Sector(pub i32);

/// SU(2) symmetry sector (2j, m)
///
/// j = 0, 1/2, 1, 3/2, ... represented as 2j = 0, 1, 2, 3, ...
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SU2Sector(pub u32, pub i32);

/// General sector for custom symmetries
///
/// Multi-dimensional charge vector
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sector(pub Vec<i32>);
