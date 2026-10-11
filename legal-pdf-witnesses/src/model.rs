//! Decisions learned from the structure gold: gradient-boosted trees (LightGBM's JSON dump) over the
//! witnesses. The unit model reads each counted line with the lines before and after it and names
//! the unit it opens (none, heading, list item, numbered paragraph); the parent model scores an
//! outline unit's candidate parents. Both ship in `data/` (tools: the structure gold's trainers);
//! `LEGALPDF_UNIT_MODEL` and `LEGALPDF_PARENT_MODEL` name others, and `LEGALPDF_STRUCTURE_MODELS=off`
//! leaves the readers' own decisions (the Rust-only tier).

use crate::witness::Table;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Deserialize)]
#[serde(untagged)]
enum Node {
    Split { split_feature: usize, threshold: f64, left_child: Box<Node>, right_child: Box<Node> },
    Leaf { leaf_value: f64 },
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
    none_below: Option<f64>,
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
    /// A ranker's score below which no candidate is taken for a unit on a recognized line (it stands
    /// at the top level).
    pub none_below: Option<f64>,
}

fn flatten(node: &Node, splits: &mut Vec<Split>, leaves: &mut Vec<f64>) -> i32 {
    match node {
        Node::Leaf { leaf_value } => {
            leaves.push(*leaf_value);
            -(leaves.len() as i32)
        }
        Node::Split { split_feature, threshold, left_child, right_child } => {
            let at = splits.len();
            splits.push(Split { feature: *split_feature as u32, threshold: *threshold, left: 0, right: 0 });
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
        let roots = dump.tree_info.iter().map(|tree| flatten(&tree.tree_structure, &mut splits, &mut leaves)).collect();
        Model { num_class: dump.num_class, roots, splits, leaves, none_below: dump.none_below }
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
                at = if f64::from(features[split.feature as usize]) <= split.threshold { split.left } else { split.right };
            }
            scores[n % self.num_class] += self.leaves[(-at - 1) as usize];
        }
        scores
    }
}

const UNIT: &str = include_str!("../data/unit_model.json");
const PARENT: &str = include_str!("../data/parent_model.json");

/// A model's text: the file its variable names, else the shipped one; none where the models are off.
fn text(variable: &str, shipped: &'static str) -> Option<String> {
    if std::env::var("LEGALPDF_STRUCTURE_MODELS").is_ok_and(|v| v == "off") {
        return None;
    }
    Some(std::env::var(variable).ok().and_then(|path| std::fs::read_to_string(path).ok()).unwrap_or_else(|| shipped.to_owned()))
}

/// The learned models in force, as text, for the caches of what they decide.
pub fn models_text() -> String {
    [text("LEGALPDF_UNIT_MODEL", UNIT), text("LEGALPDF_PARENT_MODEL", PARENT)].map(Option::unwrap_or_default).join("\n")
}

fn load(variable: &str, shipped: &'static str) -> Option<Model> {
    serde_json::from_str::<Dump>(&text(variable, shipped)?).ok().map(Model::from)
}

pub fn unit_model() -> Option<&'static Model> {
    static LOADED: OnceLock<Option<Model>> = OnceLock::new();
    LOADED.get_or_init(|| load("LEGALPDF_UNIT_MODEL", UNIT)).as_ref()
}

pub fn parent_model() -> Option<&'static Model> {
    static LOADED: OnceLock<Option<Model>> = OnceLock::new();
    LOADED.get_or_init(|| load("LEGALPDF_PARENT_MODEL", PARENT)).as_ref()
}

/// The unit kinds the unit model names.
pub const UNIT_KINDS: &[&str] = &["none", "heading", "list_item", "numbered_paragraph"];

/// Each counted line's unit kind by the unit model, where one is in force: the line's witnesses, then
/// the line before's, then the line after's, in the table's reading order.
pub fn unit_kinds(table: &Table) -> Option<HashMap<String, u8>> {
    let model = unit_model()?;
    let width = table.rows.first().map_or(0, |row| row.values.len());
    let zero = vec![0.0; width];
    let mut features = Vec::with_capacity(width * 3);
    Some(
        table
            .rows
            .iter()
            .enumerate()
            .map(|(n, row)| {
                features.clear();
                features.extend_from_slice(&row.values);
                features.extend_from_slice(n.checked_sub(1).map_or(&zero, |m| &table.rows[m].values));
                features.extend_from_slice(table.rows.get(n + 1).map_or(&zero, |next| &next.values));
                let scores = model.scores(&features);
                let kind = (0..scores.len()).max_by(|a, b| scores[*a].total_cmp(&scores[*b])).unwrap_or(0);
                (row.id.clone(), kind as u8)
            })
            .collect(),
    )
}
