//! A forest of decision trees as LightGBM writes one (its text model), read and evaluated here: the
//! scorer's tables, fitted on the gold and run in pure Rust, deterministic, with no runtime. A
//! multiclass forest's trees take the classes in turn; its scores are each class's summed leaves.
//! Only numerical splits are read (the scorer's features are numbers); a missing value is none.

use std::path::Path;

struct Tree {
    feature: Vec<u32>,
    threshold: Vec<f64>,
    default_left: Vec<bool>,
    left: Vec<i32>,
    right: Vec<i32>,
    leaf: Vec<f64>,
}

impl Tree {
    fn score(&self, x: &[f32]) -> f64 {
        if self.feature.is_empty() {
            return self.leaf[0];
        }
        let mut node = 0i32;
        while node >= 0 {
            let n = node as usize;
            let value = f64::from(x[self.feature[n] as usize]);
            let left = if value.is_nan() { self.default_left[n] } else { value <= self.threshold[n] };
            node = if left { self.left[n] } else { self.right[n] };
        }
        self.leaf[(!node) as usize]
    }
}

pub(crate) struct Forest {
    classes: usize,
    trees: Vec<Tree>,
}

impl Forest {
    /// The forest in `path`, which must read at most `width` features and score `expected` classes.
    pub(crate) fn read(path: &Path, width: usize, expected: usize) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut classes = 1;
        let mut features = 0usize;
        let mut trees = Vec::new();
        let mut fields: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        let mut in_tree = false;
        let finish = |fields: &std::collections::HashMap<&str, &str>| -> Result<Tree, String> {
            let numbers = |key: &str| -> Result<Vec<f64>, String> {
                fields.get(key).map_or(Ok(Vec::new()), |v| v.split_whitespace().map(|t| t.parse::<f64>().map_err(|e| format!("{key}: {e}"))).collect())
            };
            let integers = |key: &str| -> Result<Vec<i64>, String> {
                fields.get(key).map_or(Ok(Vec::new()), |v| v.split_whitespace().map(|t| t.parse::<i64>().map_err(|e| format!("{key}: {e}"))).collect())
            };
            Ok(Tree {
                feature: integers("split_feature")?.into_iter().map(|f| f as u32).collect(),
                threshold: numbers("threshold")?,
                // decision_type bit 1: a missing value goes left.
                default_left: integers("decision_type")?.into_iter().map(|d| d & 2 != 0).collect(),
                left: integers("left_child")?.into_iter().map(|c| c as i32).collect(),
                right: integers("right_child")?.into_iter().map(|c| c as i32).collect(),
                leaf: numbers("leaf_value")?,
            })
        };
        for line in text.lines() {
            if let Some(n) = line.strip_prefix("num_class=") {
                classes = n.trim().parse().map_err(|e| format!("num_class: {e}"))?;
            } else if let Some(n) = line.strip_prefix("max_feature_idx=") {
                features = n.trim().parse::<usize>().map_err(|e| format!("max_feature_idx: {e}"))? + 1;
            } else if line.starts_with("Tree=") {
                if in_tree {
                    trees.push(finish(&fields)?);
                }
                fields.clear();
                in_tree = true;
            } else if line.starts_with("end of trees") {
                if in_tree {
                    trees.push(finish(&fields)?);
                }
                break;
            } else if in_tree {
                if let Some((key, value)) = line.split_once('=') {
                    fields.insert(key, value);
                }
            }
        }
        if trees.is_empty() {
            return Err(format!("{}: no trees", path.display()));
        }
        if features > width || classes != expected {
            return Err(format!("{}: {features} features and {classes} classes, where the parser gives {width} and scores {expected}", path.display()));
        }
        Ok(Self { classes, trees })
    }

    /// Each class's raw score for `x` (one score for a binary forest).
    pub(crate) fn scores(&self, x: &[f32]) -> Vec<f64> {
        let mut out = vec![0.0; self.classes];
        for (k, tree) in self.trees.iter().enumerate() {
            out[k % self.classes] += tree.score(x);
        }
        out
    }

    /// A multiclass forest's class probabilities for `x` (the softmax of its scores).
    pub(crate) fn probabilities(&self, x: &[f32]) -> Vec<f64> {
        let scores = self.scores(x);
        let top = scores.iter().copied().fold(f64::MIN, f64::max);
        let exp: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
        let total: f64 = exp.iter().sum();
        exp.into_iter().map(|e| e / total).collect()
    }
}
