//! Lightweight native evaluation of offline-trained LightGBM trees.
//! The source-span model predicts boundaries and eight primary roles; the attachment model scores
//! the root and open ancestors. LEGALPDF_SPAN_MODEL and LEGALPDF_PARENT_MODEL supply compatible
//! weights; LEGALPDF_STRUCTURE_MODELS=off disables learned scoring. Historical line-key/rootless
//! weights cannot satisfy these feature contracts.

use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
#[serde(untagged)]
enum Node {
    Split {
        split_feature: usize,
        threshold: f64,
        left_child: Box<Node>,
        right_child: Box<Node>,
    },
    Leaf {
        leaf_value: f64,
    },
}

#[derive(Deserialize)]
struct Tree {
    tree_structure: Node,
}

#[derive(Deserialize)]
struct Dump {
    #[serde(default = "one")]
    num_class: usize,
    tree_info: Vec<Tree>,
    #[serde(default)]
    feature_schema: Option<String>,
}

/// One split, its children as indexes into the nodes, or (negated, less one) into the leaves.
struct Split {
    feature: u32,
    threshold: f64,
    left: i32,
    right: i32,
}

/// The trees flattened into one array of splits and one of leaves, each tree a root index.
pub struct Model {
    num_class: usize,
    roots: Vec<i32>,
    splits: Vec<Split>,
    leaves: Vec<f64>,
}

fn flatten(node: &Node, splits: &mut Vec<Split>, leaves: &mut Vec<f64>) -> i32 {
    match node {
        Node::Leaf { leaf_value } => {
            leaves.push(*leaf_value);
            -(leaves.len() as i32)
        }
        Node::Split {
            split_feature,
            threshold,
            left_child,
            right_child,
        } => {
            let at = splits.len();
            splits.push(Split {
                feature: *split_feature as u32,
                threshold: *threshold,
                left: 0,
                right: 0,
            });
            let left = flatten(left_child, splits, leaves);
            let right = flatten(right_child, splits, leaves);
            splits[at].left = left;
            splits[at].right = right;
            at as i32
        }
    }
}

impl From<Dump> for Model {
    fn from(dump: Dump) -> Self {
        let (mut splits, mut leaves) = (Vec::new(), Vec::new());
        let roots = dump
            .tree_info
            .iter()
            .map(|tree| flatten(&tree.tree_structure, &mut splits, &mut leaves))
            .collect();
        Model {
            num_class: dump.num_class,
            roots,
            splits,
            leaves,
        }
    }
}

fn one() -> usize {
    1
}

impl Model {
    /// The raw score of each class (one for a ranker).
    pub fn scores(&self, features: &[f32]) -> Vec<f64> {
        let mut scores = vec![0.0; self.num_class];
        for (n, &root) in self.roots.iter().enumerate() {
            let mut at = root;
            while at >= 0 {
                let split = &self.splits[at as usize];
                at = if f64::from(features[split.feature as usize]) <= split.threshold {
                    split.left
                } else {
                    split.right
                };
            }
            scores[n % self.num_class] += self.leaves[(-at - 1) as usize];
        }
        scores
    }
}

const PARENT: &str = include_str!("../data/parent_model.json");

/// A model's text: the file its variable names, else the shipped one; none where the models are off.
fn text(variable: &str, shipped: &'static str) -> Option<String> {
    if std::env::var("LEGALPDF_STRUCTURE_MODELS").is_ok_and(|v| v == "off") {
        return None;
    }
    Some(
        std::env::var(variable)
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .unwrap_or_else(|| shipped.to_owned()),
    )
}

/// The learned models in force, as text, for the caches of what they decide.
pub fn models_text() -> String {
    [
        text("LEGALPDF_SPAN_MODEL", ""),
        text("LEGALPDF_SCOPE_MODEL", ""),
        text("LEGALPDF_SCOPE_OPENING_MODEL", ""),
        text("LEGALPDF_MEMBERSHIP_MODEL", ""),
        text("LEGALPDF_PARENT_MODEL", PARENT),
    ]
    .map(Option::unwrap_or_default)
    .join("\n")
}

pub fn span_model() -> Option<&'static Model> {
    static LOADED: OnceLock<Option<Model>> = OnceLock::new();
    LOADED
        .get_or_init(|| {
            let dump: Dump = serde_json::from_str(&text("LEGALPDF_SPAN_MODEL", "")?).ok()?;
            if dump.feature_schema.as_deref() != Some(crate::spans::SPAN_SCHEMA)
                || dump.num_class != crate::spans::ROLE_NAMES.len()
            {
                return None;
            }
            let model = Model::from(dump);
            let width = crate::spans::feature_names().len();
            model
                .splits
                .iter()
                .all(|split| (split.feature as usize) < width)
                .then_some(model)
        })
        .as_ref()
}

pub fn parent_model() -> Option<&'static Model> {
    static LOADED: OnceLock<Option<Model>> = OnceLock::new();
    LOADED
        .get_or_init(|| {
            let dump: Dump = serde_json::from_str(&text("LEGALPDF_PARENT_MODEL", PARENT)?).ok()?;
            if dump.feature_schema.as_deref() != Some(crate::outline::PARENT_SCHEMA)
                || dump.num_class != 1
            {
                return None;
            }
            let model = Model::from(dump);
            let width = crate::outline::PAIR_NAMES.len() + 2 * crate::witness::NAMES.len();
            model
                .splits
                .iter()
                .all(|split| (split.feature as usize) < width)
                .then_some(model)
        })
        .as_ref()
}


/// Finite, nonnegative rounded counts; this head never reads authored scope identities.
pub fn scope_opening_model() -> Option<&'static Model> {
    static LOADED: OnceLock<Option<Model>> = OnceLock::new();
    LOADED.get_or_init(|| {
        let dump: Dump = serde_json::from_str(&text("LEGALPDF_SCOPE_OPENING_MODEL", "")?).ok()?;
        if dump.feature_schema.as_deref() != Some(crate::scopes::SCOPE_OPENING_SCHEMA) || dump.num_class != 1 {
            return None;
        }
        let model = Model::from(dump);
        let width = crate::scopes::opening_feature_names().len();
        model.splits.iter().all(|split| (split.feature as usize) < width).then_some(model)
    }).as_ref()
}

pub fn scope_model() -> Option<&'static Model> {
    static LOADED: OnceLock<Option<Model>> = OnceLock::new();
    LOADED
        .get_or_init(|| {
            let dump: Dump = serde_json::from_str(&text("LEGALPDF_SCOPE_MODEL", "")?).ok()?;
            if dump.feature_schema.as_deref() != Some(crate::scopes::SCOPE_SCHEMA)
                || dump.num_class != 1
            {
                return None;
            }
            let model = Model::from(dump);
            let width = crate::scopes::feature_names().len();
            model
                .splits
                .iter()
                .all(|split| (split.feature as usize) < width)
                .then_some(model)
        })
        .as_ref()
}

pub fn membership_model() -> Option<&'static Model> {
    static LOADED: OnceLock<Option<Model>> = OnceLock::new();
    LOADED
        .get_or_init(|| {
            let dump: Dump = serde_json::from_str(&text("LEGALPDF_MEMBERSHIP_MODEL", "")?).ok()?;
            if dump.feature_schema.as_deref() != Some(crate::membership::MEMBERSHIP_SCHEMA)
                || dump.num_class != 1
            {
                return None;
            }
            let model = Model::from(dump);
            let width = crate::membership::feature_names().len();
            model
                .splits
                .iter()
                .all(|split| (split.feature as usize) < width)
                .then_some(model)
        })
        .as_ref()
}
