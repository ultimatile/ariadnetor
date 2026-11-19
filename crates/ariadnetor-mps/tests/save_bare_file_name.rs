//! Saving to a path with no directory component.
//!
//! The test changes the process's working directory, so it lives in a test
//! binary of its own: no other test in this binary may depend on the
//! working directory.

use ariadnetor_mps::{Mps, TensorChain, load_mps_from_path, save_mps_to_path};
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensorData, MemoryOrder, Tensor};

/// A bare file name saves into the working directory, temp file included,
/// and leaves only the destination there.
#[test]
fn save_to_a_bare_file_name_writes_into_the_working_directory() {
    let dir = std::env::temp_dir().join(format!("ariadnetor_mps_bare_name_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    std::env::set_current_dir(&dir).expect("enter scratch dir");

    let mps: Mps<DenseStorage<f64>, DenseLayout> = Mps::from_sites(vec![Tensor::from_data(
        DenseTensorData::from_raw_parts(vec![0.6, 0.8], vec![1, 2, 1], MemoryOrder::ColumnMajor),
    )]);
    save_mps_to_path(&mps, "state.mps").expect("save to a bare file name");
    let loaded: Mps<DenseStorage<f64>, DenseLayout> =
        load_mps_from_path("state.mps").expect("load from a bare file name");
    assert_eq!(loaded.site(0).data_slice(), mps.site(0).data_slice());

    let entries: Vec<String> = std::fs::read_dir(".")
        .expect("read the scratch dir")
        .map(|e| e.expect("dir entry").file_name().into_string().unwrap())
        .collect();
    assert_eq!(entries, ["state.mps"], "temp files left behind");
    std::fs::remove_dir_all(&dir).expect("remove scratch dir");
}
