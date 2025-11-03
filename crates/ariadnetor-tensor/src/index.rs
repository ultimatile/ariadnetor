//! Index and IndexSet for tensor metadata

use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique IndexIds
static INDEX_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for an Index
///
/// Each Index gets a globally unique ID to distinguish it from other indices,
/// even if they have the same tags and prime level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexId(u64);

impl IndexId {
    /// Generate a new unique IndexId
    fn generate() -> Self {
        IndexId(INDEX_ID_COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

/// Index represents a tensor dimension with metadata
///
/// # Fields
///
/// - `id`: Unique identifier for matching indices
/// - `dim`: Dimension size (0 means unspecified, determined at tensor creation)
/// - `tags`: String tags for semantic identification (e.g., "Site", "Link")
/// - `prime_level`: Prime level for index notation (i, i', i'', etc.)
#[derive(Debug, Clone)]
pub struct Index {
    pub id: IndexId,
    pub dim: usize,
    pub tags: Vec<String>,
    pub prime_level: u8,
}

impl Index {
    /// Create a new Index with a tag and unspecified dimension
    ///
    /// # Arguments
    ///
    /// * `tag` - Tag string for semantic identification
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let idx = Index::new("i");
    /// ```
    pub fn new(tag: &str) -> Self {
        Self {
            id: IndexId::generate(),
            dim: 0, // Unspecified, will be set when attached to tensor
            tags: vec![tag.to_string()],
            prime_level: 0,
        }
    }

    /// Create a new Index with a tag and specified dimension
    ///
    /// # Arguments
    ///
    /// * `tag` - Tag string for semantic identification
    /// * `dim` - Dimension size
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let idx = Index::with_dim("i", 10);
    /// ```
    pub fn with_dim(tag: &str, dim: usize) -> Self {
        Self {
            id: IndexId::generate(),
            dim,
            tags: vec![tag.to_string()],
            prime_level: 0,
        }
    }

    /// Check if this index has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Add a tag to this index
    pub fn add_tag(&mut self, tag: &str) {
        if !self.has_tag(tag) {
            self.tags.push(tag.to_string());
        }
    }

    /// Create a primed copy of this index (i → i')
    pub fn prime(&self) -> Self {
        Self {
            id: IndexId::generate(), // New ID for primed index
            dim: self.dim,
            tags: self.tags.clone(),
            prime_level: self.prime_level + 1,
        }
    }
}

impl PartialEq for Index {
    /// Two indices are equal if they have the same ID
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Index {}

/// Set of indices with row/column rank information
///
/// # Fields
///
/// - `indices`: List of indices for each tensor dimension
/// - `rowrank`: Number of "row" indices (for matrix-like operations)
#[derive(Debug, Clone, PartialEq)]
pub struct IndexSet {
    pub indices: Vec<Index>,
    pub rowrank: usize,
}

impl IndexSet {
    /// Create a new IndexSet
    pub fn new(indices: Vec<Index>, rowrank: usize) -> Self {
        Self { indices, rowrank }
    }
}
