//! Lightweight label identifier (interned string)

use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, Ordering};

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LabelId(u64);

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
            next_id: AtomicU64::new(1),
        }
    }
}

impl LabelId {
    pub fn intern(name: &str) -> Self {
        if let Some(id) = INTERNER.name_to_id.get(name) {
            return LabelId(*id);
        }
        let id = INTERNER.next_id.fetch_add(1, Ordering::SeqCst);
        INTERNER.name_to_id.insert(name.to_string(), id);
        INTERNER.id_to_name.insert(id, name.to_string());
        LabelId(id)
    }

    pub fn name(&self) -> String {
        INTERNER
            .id_to_name
            .get(&self.0)
            .map(|e| e.value().clone())
            .unwrap_or_else(|| format!("<unknown:{}>", self.0))
    }

    pub fn fresh() -> Self {
        let id = INTERNER.next_id.fetch_add(1, Ordering::SeqCst);
        let name = format!("_tmp_{}", id);
        INTERNER.name_to_id.insert(name.clone(), id);
        INTERNER.id_to_name.insert(id, name);
        LabelId(id)
    }

    pub fn prime(&self) -> Self {
        Self::intern(&format!("{}'", self.name()))
    }

    pub fn primes(&self, n: usize) -> Self {
        Self::intern(&format!("{}{}", self.name(), "'".repeat(n)))
    }

    pub fn unprime(&self) -> Self {
        let name = self.name();
        if let Some(stripped) = name.strip_suffix('\'') {
            Self::intern(stripped)
        } else {
            *self
        }
    }

    pub fn base(&self) -> Self {
        Self::intern(self.name().trim_end_matches('\''))
    }
}

#[macro_export]
macro_rules! label {
    ($name:expr) => {
        $crate::LabelId::intern($name)
    };
}

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
        let i = LabelId::intern("test_i");
        assert_eq!(i.prime().name(), "test_i'");
        assert_eq!(i.primes(2).name(), "test_i''");
    }

    #[test]
    fn test_fresh_unique() {
        let f1 = LabelId::fresh();
        let f2 = LabelId::fresh();
        assert_ne!(f1, f2);
    }
}
