//! Sealed [`MpsCodec`] dispatch over the supported `(Storage, Layout)` pairs.
//!
//! The trait resolves the otherwise-uninferrable free scalar parameter of
//! `save_mps` / `load_mps` by binding it to the storage element type, and
//! routes each site through the tensor crate's per-tensor codec. It is sealed
//! via a crate-private supertrait, so the supported pairs are closed.

use ariadnetor_tensor::{
    BlockSparseLayout, BlockSparseStorage, BodyMeta, DenseLayout, DenseStorage, ScalarCodec,
    SectorTag, SerializableSector, StorageTag, Tensor, decode_block_sparse, decode_dense,
    encode_block_sparse, encode_dense,
};

use super::error::MpsIoError;
use super::manifest::SiteMeta;
use crate::{CanonicalForm, Mps, TensorChain};

mod sealed {
    use ariadnetor_tensor::{
        BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, ScalarCodec,
        SerializableSector,
    };

    use crate::Mps;

    pub trait Sealed {}
    impl<T: ScalarCodec> Sealed for Mps<DenseStorage<T>, DenseLayout> {}
    impl<T: ScalarCodec, S: SerializableSector> Sealed
        for Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>
    {
    }
}

/// Storage-keyed serialization dispatch for an `Mps`.
///
/// Implemented for the dense and block-sparse chain shapes; the block-sparse
/// impl additionally carries the sector codec. See the module-level
/// documentation.
pub trait MpsCodec: sealed::Sealed + Sized {
    /// The stored scalar element type.
    type Scalar: ScalarCodec;

    /// Storage-kind tag written into the manifest.
    const STORAGE_TAG: StorageTag;

    /// Sector-type tag, or `None` for a sectorless (dense) chain.
    fn sector_tag() -> Option<SectorTag>;

    /// Encode every site into a descriptor table plus the concatenated
    /// numeric data section.
    fn encode_sites(&self) -> (Vec<SiteMeta>, Vec<u8>);

    /// Reconstruct the chain from the manifest's descriptor table and the
    /// numeric data section.
    fn decode_sites(
        sites: &[SiteMeta],
        data: &[u8],
        canonical: CanonicalForm,
    ) -> Result<Self, MpsIoError>;
}

/// Slice the next site's bytes off the data section, advancing `offset`.
fn site_slice<'a>(
    data: &'a [u8],
    offset: &mut usize,
    data_len: u64,
) -> Result<&'a [u8], MpsIoError> {
    let len = usize::try_from(data_len).map_err(|_| MpsIoError::Corrupt {
        detail: "site data length exceeds usize".to_string(),
    })?;
    let end = offset.checked_add(len).ok_or(MpsIoError::Corrupt {
        detail: "data-section offset overflow".to_string(),
    })?;
    let slice = data.get(*offset..end).ok_or(MpsIoError::UnexpectedEof)?;
    *offset = end;
    Ok(slice)
}

/// Assemble decoded sites into an `Mps`, rejecting a data section that was not
/// fully consumed.
fn assemble<St, L>(
    tensors: Vec<Tensor<St, L>>,
    consumed: usize,
    data_len: usize,
    canonical: CanonicalForm,
) -> Result<Mps<St, L>, MpsIoError>
where
    St: ariadnetor_tensor::Storage + ariadnetor_tensor::StorageFor<L>,
    L: ariadnetor_tensor::TensorLayout,
{
    if consumed != data_len {
        return Err(MpsIoError::Corrupt {
            detail: "trailing bytes in data section".to_string(),
        });
    }
    let mut mps = if tensors.is_empty() {
        Mps::empty()
    } else {
        Mps::from_sites(tensors)
    };
    mps.set_canonical_form(canonical);
    Ok(mps)
}

impl<T: ScalarCodec> MpsCodec for Mps<DenseStorage<T>, DenseLayout> {
    type Scalar = T;
    const STORAGE_TAG: StorageTag = StorageTag::Dense;

    fn sector_tag() -> Option<SectorTag> {
        None
    }

    fn encode_sites(&self) -> (Vec<SiteMeta>, Vec<u8>) {
        let mut sites = Vec::with_capacity(self.len());
        let mut data = Vec::new();
        for site in self.sites() {
            let td = site.data();
            let (body, bytes) = encode_dense(td);
            let data_len = bytes.len() as u64;
            data.extend_from_slice(&bytes);
            sites.push(SiteMeta {
                order: td.order().into(),
                body,
                data_len,
            });
        }
        (sites, data)
    }

    fn decode_sites(
        sites: &[SiteMeta],
        data: &[u8],
        canonical: CanonicalForm,
    ) -> Result<Self, MpsIoError> {
        let mut tensors = Vec::with_capacity(sites.len());
        let mut offset = 0usize;
        for meta in sites {
            let slice = site_slice(data, &mut offset, meta.data_len)?;
            let shape = match &meta.body {
                BodyMeta::Dense { shape } => shape,
                BodyMeta::BlockSparse { .. } => {
                    return Err(MpsIoError::Corrupt {
                        detail: "block-sparse site in a dense chain".to_string(),
                    });
                }
            };
            let td = decode_dense::<T>(shape, meta.order.into(), slice)?;
            tensors.push(Tensor::from_data(td));
        }
        assemble(tensors, offset, data.len(), canonical)
    }
}

impl<T: ScalarCodec, S: SerializableSector> MpsCodec
    for Mps<BlockSparseStorage<T>, BlockSparseLayout<S>>
{
    type Scalar = T;
    const STORAGE_TAG: StorageTag = StorageTag::BlockSparse;

    fn sector_tag() -> Option<SectorTag> {
        Some(S::type_tag())
    }

    fn encode_sites(&self) -> (Vec<SiteMeta>, Vec<u8>) {
        let mut sites = Vec::with_capacity(self.len());
        let mut data = Vec::new();
        for site in self.sites() {
            let td = site.data();
            let (body, bytes) = encode_block_sparse(td);
            let data_len = bytes.len() as u64;
            data.extend_from_slice(&bytes);
            sites.push(SiteMeta {
                order: td.layout().order().into(),
                body,
                data_len,
            });
        }
        (sites, data)
    }

    fn decode_sites(
        sites: &[SiteMeta],
        data: &[u8],
        canonical: CanonicalForm,
    ) -> Result<Self, MpsIoError> {
        let mut tensors = Vec::with_capacity(sites.len());
        let mut offset = 0usize;
        for meta in sites {
            let slice = site_slice(data, &mut offset, meta.data_len)?;
            let (flux, indices) = match &meta.body {
                BodyMeta::BlockSparse { flux, indices } => (flux, indices),
                BodyMeta::Dense { .. } => {
                    return Err(MpsIoError::Corrupt {
                        detail: "dense site in a block-sparse chain".to_string(),
                    });
                }
            };
            let td = decode_block_sparse::<T, S>(flux, indices, meta.order.into(), slice)?;
            tensors.push(Tensor::from_data(td));
        }
        assemble(tensors, offset, data.len(), canonical)
    }
}
