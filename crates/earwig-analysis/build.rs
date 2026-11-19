// The prebuilt ONNX Runtime used by `ort` calls `__isPlatformVersionAtLeast` (compiler-rt).
// rustc links with -nodefaultlibs, so on macOS link clang's runtime explicitly.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = std::process::Command::new("clang")
        .arg("--print-resource-dir")
        .output();
    if let Ok(o) = out {
        let dir = String::from_utf8_lossy(&o.stdout).trim().to_string();
        println!("cargo:rustc-link-search=native={dir}/lib/darwin");
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}
