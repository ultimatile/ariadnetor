//! Index and IndexSet for tensor metadata

/// Unique identifier for an Index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexId(u64);

/// Index represents a tensor dimension with metadata
#[derive(Debug, Clone)]
pub struct Index {
    pub id: IndexId,
    pub dim: usize,
    pub tags: Vec<String>,
    pub prime_level: u8,
}

impl Index {
    /// Create a new Index with a tag
    pub fn new(_tag: &str) -> Self {
        // TODO: Implement proper IndexId generation
        unimplemented!("Index creation not yet implemented")
    }
}

/// Set of indices with row/column rank information
#[derive(Debug, Clone)]
pub struct IndexSet {
    pub indices: Vec<Index>,
    pub rowrank: usize,
}
