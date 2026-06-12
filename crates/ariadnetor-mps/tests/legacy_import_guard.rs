//! Source-scan guard: production code in this crate must reach linalg
//! operations through the explicit-backend `*_with_backend` twins, never
//! the legacy wrappers that derive the backend from a tensor argument.
//! A legacy call site needs an import, a qualified `arnet::`-path
//! reference (call, turbofish, or function value), or a crate alias;
//! this scan rejects all three forms, so reintroducing one fails the
//! suite instead of relying on review. The guard is transitional: it
//! ends with the legacy wrappers themselves, whose removal deletes
//! every name below.
//!
//! This file is the single source; the other guarded crate compiles it
//! via a `#[path]` include, so the `env!`-based paths resolve against
//! whichever crate is running the test.

use std::fs;
use std::path::Path;

/// The legacy backend-derived wrappers exported by the linalg crate
/// (base operations plus the `*_with_policy` expert variants). The
/// `*_with_backend` twins and unrelated identifiers that merely contain
/// these names as substrings do not match — the scan compares whole
/// identifier tokens.
const FORBIDDEN: [&str; 36] = [
    "contract",
    "contract_block_sparse",
    "contract_with_policy",
    "diag",
    "diagonal_scale",
    "diagonal_scale_block_sparse",
    "eig",
    "eig_with_policy",
    "eigh",
    "eigh_with_policy",
    "eigvals",
    "eigvalsh",
    "einsum",
    "expm",
    "expm_antihermitian",
    "expm_hermitian",
    "fuse_legs_block_sparse",
    "inverse",
    "lq",
    "lq_block_sparse",
    "lq_with_policy",
    "permute_block_sparse",
    "qr",
    "qr_block_sparse",
    "qr_with_policy",
    "solve",
    "solve_with_policy",
    "svd",
    "svd_block_sparse",
    "svd_with_policy",
    "trace",
    "transpose",
    "transpose_with_policy",
    "trunc_svd",
    "trunc_svd_block_sparse",
    "trunc_svd_with_policy",
];

/// Paths whose references are scanned: the umbrella plus the linalg
/// leaf crate (where the legacy wrappers are defined), so a future
/// direct leaf dependency cannot silently reopen the legacy surface.
const SCANNED_ROOTS: [&str; 2] = ["arnet", "arnet_linalg"];

fn visit(dir: &Path, hits: &mut Vec<String>) {
    for entry in fs::read_dir(dir).expect("readable src directory") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            visit(&path, hits);
        } else if path.extension().is_some_and(|e| e == "rs") {
            scan_file(&path, hits);
        }
    }
}

/// Strip a `//` line comment so prose mentioning a legacy name does not
/// trip the token scan. Token-based, not a lexer: a string literal
/// containing `//` truncates the rest of its line (accepted residual —
/// the scanned crates' production code carries no such literals).
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(pos) => &line[..pos],
        None => line,
    }
}

/// Skip block-comment-style lines (`/* ...` or the conventional ` * `
/// continuation) so commented-out prose does not trip the token scan.
fn is_block_comment_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("/*") || trimmed.starts_with('*')
}

fn tokens(text: &str) -> Vec<&str> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .collect()
}

/// A crate-level alias (`use arnet as ar;` / `use arnet::{self as ar}`)
/// would let later references evade the qualified-path scan, so
/// aliasing a scanned root is rejected outright.
fn aliases_scanned_root(decl_tokens: &[&str]) -> bool {
    decl_tokens.iter().enumerate().any(|(i, tok)| {
        SCANNED_ROOTS.contains(tok)
            && (decl_tokens.get(i + 1) == Some(&"as")
                || (decl_tokens.get(i + 1) == Some(&"self")
                    && decl_tokens.get(i + 2) == Some(&"as")))
    })
}

fn scan_file(path: &Path, hits: &mut Vec<String>) {
    let src = fs::read_to_string(path).expect("readable source file");

    // Join multi-line `use` declarations (any visibility) so brace
    // lists scan as one unit, and scan each line's token stream for
    // qualified references (`arnet::<legacy-name>` in any position —
    // call, turbofish, or function value).
    let mut current: Option<(usize, String)> = None;
    for (idx, raw_line) in src.lines().enumerate() {
        if is_block_comment_line(raw_line) {
            continue;
        }
        let line = strip_line_comment(raw_line);
        let trimmed = line.trim_start();
        let is_use_start = trimmed.starts_with("use ")
            || (trimmed.starts_with("pub") && trimmed.contains(" use "));
        if current.is_none() && is_use_start {
            current = Some((idx + 1, String::new()));
        }
        if let Some((_, buf)) = current.as_mut() {
            buf.push_str(line);
            buf.push(' ');
            if line.contains(';') {
                let (lineno, decl) = current.take().expect("active declaration");
                check_decl(path, lineno, &decl, hits);
            }
        }

        let toks = tokens(line);
        for pair in toks.windows(2) {
            if SCANNED_ROOTS.contains(&pair[0]) && FORBIDDEN.contains(&pair[1]) {
                hits.push(format!(
                    "{}:{}: qualified legacy reference `{}::{}`",
                    path.display(),
                    idx + 1,
                    pair[0],
                    pair[1],
                ));
            }
        }
        // A raw identifier (`arnet::r#contract`) tokenizes as
        // `arnet`, `r`, `<name>`; catch that spelling too.
        for triple in toks.windows(3) {
            if SCANNED_ROOTS.contains(&triple[0])
                && triple[1] == "r"
                && FORBIDDEN.contains(&triple[2])
            {
                hits.push(format!(
                    "{}:{}: qualified legacy reference `{}::r#{}`",
                    path.display(),
                    idx + 1,
                    triple[0],
                    triple[2],
                ));
            }
        }
        // `extern crate arnet as x;` would alias a scanned root past
        // both the import scan and the qualified-reference scan; the
        // 2018+ editions never need the form, so reject it wholesale.
        for triple in toks.windows(3) {
            if triple[0] == "extern" && triple[1] == "crate" && SCANNED_ROOTS.contains(&triple[2]) {
                hits.push(format!(
                    "{}:{}: extern-crate declaration of a scanned crate defeats the scan",
                    path.display(),
                    idx + 1,
                ));
            }
        }
    }
}

fn check_decl(path: &Path, lineno: usize, decl: &str, hits: &mut Vec<String>) {
    let toks = tokens(decl);
    if !SCANNED_ROOTS.iter().any(|root| toks.contains(root)) {
        return;
    }
    // A glob import (`::*` or a `*` inside a brace list) would let a
    // bare legacy call escape the name scan, so it is rejected
    // outright, like a crate alias. `*` has no other meaning inside a
    // `use` declaration.
    if decl.contains('*') {
        hits.push(format!(
            "{}:{lineno}: glob import from a scanned crate defeats the legacy-name scan",
            path.display()
        ));
    }
    if aliases_scanned_root(&toks) {
        hits.push(format!(
            "{}:{lineno}: aliasing a scanned crate defeats the qualified-path scan",
            path.display()
        ));
    }
    for name in FORBIDDEN {
        if toks.contains(&name) {
            hits.push(format!(
                "{}:{lineno}: legacy import `{name}`",
                path.display()
            ));
        }
    }
}

#[test]
fn production_code_imports_no_legacy_backend_derived_ops() {
    let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut hits = Vec::new();
    visit(&src_root, &mut hits);
    assert!(
        hits.is_empty(),
        "legacy backend-derived linalg wrappers referenced from production code:\n{}",
        hits.join("\n"),
    );
}
