use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, Ordering};

/// Lightweight label identifier (interned string)
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LabelId(u64);

/// Global interner for label strings
static INTERNER: Lazy<LabelInterner> = Lazy::new(LabelInterner::default);

struct LabelInterner {
    name_to_id: DashMap<String, u64>,
    id_to_name: DashMap<u64, String>,
    next_id: AtomicU64,
}

impl Default for LabelInterner {
    fn default() -> Self {
        Self {
            name_to_id: DashMap::new(),
            id_to_name: DashMap::new(),
            next_id: AtomicU64::new(1), // 0 reserved for invalid
        }
    }
}

impl LabelId {
    /// Intern a label name, returning its ID
    pub fn intern(name: &str) -> Self {
        if let Some(id) = INTERNER.name_to_id.get(name) {
            return LabelId(*id);
        }

        let id = INTERNER.next_id.fetch_add(1, Ordering::SeqCst);
        INTERNER.name_to_id.insert(name.to_string(), id);
        INTERNER.id_to_name.insert(id, name.to_string());
        LabelId(id)
    }

    /// Get the label name (requires lookup)
    pub fn name(&self) -> String {
        INTERNER
            .id_to_name
            .get(&self.0)
            .map(|e| e.value().clone())
            .unwrap_or_else(|| format!("<unknown:{}>", self.0))
    }

    /// Create a fresh unique label
    pub fn fresh() -> Self {
        let id = INTERNER.next_id.fetch_add(1, Ordering::SeqCst);
        let name = format!("_tmp_{}", id);
        INTERNER.name_to_id.insert(name.clone(), id);
        INTERNER.id_to_name.insert(id, name);
        LabelId(id)
    }

    /// Apply prime (string manipulation sugar)
    pub fn prime(&self) -> Self {
        let name = self.name();
        Self::intern(&format!("{}'", name))
    }

    /// Apply n primes
    pub fn primes(&self, n: usize) -> Self {
        let name = self.name();
        let primed = format!("{}{}", name, "'".repeat(n));
        Self::intern(&primed)
    }

    /// Remove one prime level
    pub fn unprime(&self) -> Self {
        let name = self.name();
        if let Some(stripped) = name.strip_suffix('\'') {
            Self::intern(stripped)
        } else {
            *self
        }
    }

    /// Strip all primes, return base label
    pub fn base(&self) -> Self {
        let name = self.name();
        let base = name.trim_end_matches('\'');
        Self::intern(base)
    }

    /// Check if this label has primes
    pub fn is_primed(&self) -> bool {
        self.name().contains('\'')
    }

    /// Count prime level
    pub fn prime_level(&self) -> usize {
        self.name().chars().rev().take_while(|&c| c == '\'').count()
    }
}

/// Convenience macro for label creation
#[macro_export]
macro_rules! label {
    ($name:expr) => {
        $crate::LabelId::intern($name)
    };
}

/// Convenience macro for fresh label
#[macro_export]
macro_rules! fresh {
    () => {
        $crate::LabelId::fresh()
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_idempotent() {
        let id1 = LabelId::intern("i");
        let id2 = LabelId::intern("i");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_prime_operations() {
        let i = LabelId::intern("i");
        let i_prime = i.prime();
        let i_double = i.primes(2);

        assert_eq!(i_prime.name(), "i'");
        assert_eq!(i_double.name(), "i''");
        assert_eq!(i_prime.unprime(), i);
        assert_eq!(i_double.base(), i);
    }

    #[test]
    fn test_fresh_unique() {
        let f1 = LabelId::fresh();
        let f2 = LabelId::fresh();
        assert_ne!(f1, f2);
    }

    #[test]
    fn test_prime_level() {
        let i = LabelId::intern("i");
        assert_eq!(i.prime_level(), 0);
        assert_eq!(i.prime().prime_level(), 1);
        assert_eq!(i.primes(3).prime_level(), 3);
    }
}
