use std::env;

fn main() {
    // Only build C++ components when mlir feature is enabled
    #[cfg(feature = "mlir")]
    {
        let llvm_dir = env::var("MLIR_SYS_200_PREFIX")
            .or_else(|_| env::var("LLVM_DIR"))
            .expect("MLIR_SYS_200_PREFIX or LLVM_DIR must be set");

        let mlir_dir = format!("{}/lib/cmake/mlir", llvm_dir);
        let llvm_cmake_dir = format!("{}/lib/cmake/llvm", llvm_dir);

        // Configure and build the C++ dialect using CMake
        let dst = cmake::Config::new(".")
            .define("MLIR_DIR", &mlir_dir)
            .define("LLVM_DIR", &llvm_cmake_dir)
            .define("CMAKE_BUILD_TYPE", "Release")
            .build();

        // Link the generated libraries
        println!("cargo:rustc-link-search=native={}/lib", dst.display());
        println!("cargo:rustc-link-lib=static=MLIRTNCAPI");
        println!("cargo:rustc-link-lib=static=MLIRTNDialect");
        println!("cargo:rustc-link-lib=static=MLIRTNTransforms");

        // Link MLIR and LLVM libraries
        let llvm_lib_dir = format!("{}/lib", llvm_dir);
        println!("cargo:rustc-link-search=native={}", llvm_lib_dir);

        // Link against MLIR and LLVM shared libraries
        println!("cargo:rustc-link-lib=dylib=MLIR");
        println!("cargo:rustc-link-lib=dylib=LLVM");

        // Link C++ standard library
        println!("cargo:rustc-link-lib=c++");

        // Link zstd library (required by LLVM)
        // Try Homebrew location first
        if std::path::Path::new("/opt/homebrew/lib/libzstd.dylib").exists() {
            println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
        }
        println!("cargo:rustc-link-lib=dylib=zstd");

        // Rerun build script if CMakeLists.txt or source files change
        println!("cargo:rerun-if-changed=CMakeLists.txt");
        println!("cargo:rerun-if-changed=lib/");
        println!("cargo:rerun-if-changed=include/");
    }
}
