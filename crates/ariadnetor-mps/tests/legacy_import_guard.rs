//! Source-scan guard: production code in this crate must reach linalg
//! operations through the explicit-backend `*_with_backend` twins, never
//! the legacy wrappers that derive the backend from a tensor argument.
//! Sources are parsed with `syn`, so every reference form — import
//! (including renames, globs, and brace lists), qualified path
//! (including turbofish, function values, and raw identifiers), crate
//! alias, `extern crate`, and literal `root::name` paths inside macro
//! invocations — is resolved syntactically rather than by text
//! matching; a manifest check rejects dependency renames of the
//! scanned crates. The guard targets accidental, review-missed
//! reintroduction: a macro that assembles the path from separate
//! tokens is deliberate obfuscation and stays review territory. The
//! guard is transitional: it ends with the legacy wrappers themselves,
//! whose removal deletes every name below.
//!
//! This file is the single source; the other guarded crate compiles it
//! via a `#[path]` include, so the `env!`-based paths resolve against
//! whichever crate is running the test.

use std::fs;
use std::path::{Path, PathBuf};

use proc_macro2::{TokenStream, TokenTree};
use syn::visit::Visit;

/// The legacy backend-derived wrappers exported by the linalg crate
/// (base operations plus the `*_with_policy` expert variants). The
/// `*_with_backend` twins do not match — names are compared as whole
/// identifiers.
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

/// Crate roots whose references are scanned: the umbrella plus the
/// linalg leaf crate (where the legacy wrappers are defined), so a
/// future direct leaf dependency cannot silently reopen the legacy
/// surface.
const SCANNED_ROOTS: [&str; 2] = ["arnet", "arnet_linalg"];

/// Identifier text with any raw prefix (`r#name`) stripped.
fn ident_name(ident: &proc_macro2::Ident) -> String {
    let s = ident.to_string();
    s.strip_prefix("r#").map(str::to_owned).unwrap_or(s)
}

fn is_scanned_root(name: &str) -> bool {
    SCANNED_ROOTS.contains(&name)
}

fn is_forbidden(name: &str) -> bool {
    FORBIDDEN.contains(&name)
}

struct Scan<'a> {
    file: &'a Path,
    hits: &'a mut Vec<String>,
}

impl Scan<'_> {
    fn hit(&mut self, line: usize, msg: &str) {
        self.hits
            .push(format!("{}:{line}: {msg}", self.file.display()));
    }

    /// Walk a `use` tree. `in_scanned` is true once the tree has
    /// descended through a scanned crate root as its first segment.
    fn walk_use(&mut self, tree: &syn::UseTree, in_scanned: bool, depth: usize) {
        match tree {
            syn::UseTree::Path(p) => {
                let name = ident_name(&p.ident);
                let scanned = in_scanned || (depth == 0 && is_scanned_root(&name));
                self.walk_use(&p.tree, scanned, depth + 1);
            }
            syn::UseTree::Name(n) => {
                let name = ident_name(&n.ident);
                if in_scanned && is_forbidden(&name) {
                    self.hit(
                        n.ident.span().start().line,
                        &format!("legacy import `{name}`"),
                    );
                }
            }
            syn::UseTree::Rename(r) => {
                let original = ident_name(&r.ident);
                if in_scanned {
                    if is_forbidden(&original) {
                        self.hit(
                            r.ident.span().start().line,
                            &format!("legacy import `{original}` (renamed)"),
                        );
                    } else if original == "self" {
                        self.hit(
                            r.ident.span().start().line,
                            "crate alias of a scanned crate defeats the path scan",
                        );
                    }
                } else if depth == 0 && is_scanned_root(&original) {
                    self.hit(
                        r.ident.span().start().line,
                        "crate alias of a scanned crate defeats the path scan",
                    );
                }
            }
            syn::UseTree::Glob(g) => {
                if in_scanned {
                    self.hit(
                        g.star_token.spans[0].start().line,
                        "glob import from a scanned crate defeats the name scan",
                    );
                }
            }
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.walk_use(item, in_scanned, depth);
                }
            }
        }
    }

    /// Scan a macro invocation's token stream; macro arguments are
    /// tokens, not parsed paths, so `visit_path` does not see them.
    /// After a scanned root and `::`, every following path segment is
    /// checked, and a brace group in segment position (an embedded
    /// `use` list) is checked for forbidden identifiers wholesale.
    fn scan_tokens(&mut self, stream: TokenStream) {
        let tokens: Vec<TokenTree> = stream.into_iter().collect();
        for (i, token) in tokens.iter().enumerate() {
            match token {
                TokenTree::Group(group) => self.scan_tokens(group.stream()),
                TokenTree::Ident(ident) if is_scanned_root(&ident_name(ident)) => {
                    let root = ident_name(ident);
                    let mut j = i;
                    loop {
                        let colons = matches!(
                            (tokens.get(j + 1), tokens.get(j + 2)),
                            (Some(TokenTree::Punct(a)), Some(TokenTree::Punct(b)))
                                if a.as_char() == ':' && b.as_char() == ':'
                        );
                        if !colons {
                            break;
                        }
                        match tokens.get(j + 3) {
                            Some(TokenTree::Ident(segment)) => {
                                let name = ident_name(segment);
                                if is_forbidden(&name) {
                                    self.hit(
                                        segment.span().start().line,
                                        &format!(
                                            "qualified legacy reference `{root}::{name}` \
                                             in a macro invocation",
                                        ),
                                    );
                                }
                                j += 3;
                            }
                            Some(TokenTree::Group(group)) => {
                                self.scan_group_for_forbidden(&root, group.stream());
                                break;
                            }
                            _ => break,
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Flag any forbidden identifier inside a brace group that follows
    /// a scanned root path (an embedded `use root::{...}` list).
    fn scan_group_for_forbidden(&mut self, root: &str, stream: TokenStream) {
        for token in stream {
            match token {
                TokenTree::Group(group) => self.scan_group_for_forbidden(root, group.stream()),
                TokenTree::Ident(ident) => {
                    let name = ident_name(&ident);
                    if is_forbidden(&name) {
                        self.hit(
                            ident.span().start().line,
                            &format!(
                                "qualified legacy reference `{root}::{{.., {name}, ..}}` \
                                 in a macro invocation",
                            ),
                        );
                    }
                }
                _ => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for Scan<'_> {
    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        self.walk_use(&node.tree, false, 0);
    }

    fn visit_item_extern_crate(&mut self, node: &'ast syn::ItemExternCrate) {
        if is_scanned_root(&ident_name(&node.ident)) {
            self.hit(
                node.ident.span().start().line,
                "extern-crate declaration of a scanned crate defeats the scan",
            );
        }
    }

    fn visit_path(&mut self, node: &'ast syn::Path) {
        if node.segments.len() >= 2 && is_scanned_root(&ident_name(&node.segments[0].ident)) {
            for segment in node.segments.iter().skip(1) {
                let name = ident_name(&segment.ident);
                if is_forbidden(&name) {
                    self.hit(
                        segment.ident.span().start().line,
                        &format!(
                            "qualified legacy reference `{}::{name}`",
                            ident_name(&node.segments[0].ident),
                        ),
                    );
                }
            }
        }
        syn::visit::visit_path(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        self.scan_tokens(node.tokens.clone());
        syn::visit::visit_macro(self, node);
    }
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("readable src directory") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn production_code_references_no_legacy_backend_derived_ops() {
    let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut sources = Vec::new();
    rust_sources(&src_root, &mut sources);
    let mut hits = Vec::new();
    for path in sources {
        let src = fs::read_to_string(&path).expect("readable source file");
        let file = syn::parse_file(&src).expect("production source parses");
        let mut scan = Scan {
            file: &path,
            hits: &mut hits,
        };
        scan.visit_file(&file);
    }
    assert!(
        hits.is_empty(),
        "legacy backend-derived linalg wrappers referenced from production code:\n{}",
        hits.join("\n"),
    );
}

fn scan_manifest_for_renames(manifest: &Path, renames: &mut Vec<String>) {
    let text = fs::read_to_string(manifest).expect("readable manifest");
    for (idx, line) in text.lines().enumerate() {
        // String-level scan covering both TOML quote styles; the
        // inline and multiline dependency forms both put the
        // `package` key and its value on one line.
        let renames_scanned = line.contains("package")
            && ["ariadnetor", "ariadnetor-linalg"].iter().any(|name| {
                line.contains(&format!("\"{name}\"")) || line.contains(&format!("'{name}'"))
            });
        if renames_scanned {
            renames.push(format!(
                "{}:{}: dependency rename of a scanned crate defeats the source scan",
                manifest.display(),
                idx + 1,
            ));
        }
    }
}

/// The nearest ancestor manifest declaring `[workspace]`; a renamed
/// dependency can also live there and be inherited via
/// `name.workspace = true`.
fn workspace_root_manifest(member_dir: &Path) -> Option<PathBuf> {
    let mut dir = member_dir.parent();
    while let Some(d) = dir {
        let candidate = d.join("Cargo.toml");
        if candidate.exists()
            && fs::read_to_string(&candidate)
                .expect("readable manifest")
                .contains("[workspace]")
        {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

/// A renamed Cargo dependency (`legacy = { package = "ariadnetor" }`),
/// whether declared on the member or inherited from the workspace
/// manifest, would change the crate root the source scan keys on,
/// reopening the legacy surface under a name `SCANNED_ROOTS` does not
/// list. The canonical manifests never use a `package` key for the
/// workspace crates, so any such rename is rejected.
#[test]
fn manifests_do_not_rename_scanned_crates() {
    let member_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut renames = Vec::new();
    scan_manifest_for_renames(&member_dir.join("Cargo.toml"), &mut renames);
    let root = workspace_root_manifest(member_dir).expect("workspace root manifest found");
    scan_manifest_for_renames(&root, &mut renames);
    assert!(renames.is_empty(), "{}", renames.join("\n"));
}
