//! A tree's pair scores as the decoding reads them: each node's parent and sibling query and key (the tree graph's,
//! `docparse.TreeNodes`), and for each pair the decoding chooses among (a node against the nodes on the tree's
//! right-most path, the paper's Algorithm 1) its query-key product and its codes' biases: `docparse.tree_codes` and
//! `pair_bias`, the tables the export writes (`docparse.json`'s `trees`). The scores the fit reads, for the pairs the
//! decoding reads: a tree costs its nodes times its depth, not its nodes squared.

/// A tree's pair tables: each code's row (a parent and a sibling bias), each table's first row, and the left-edge and
/// size scales.
#[derive(serde::Deserialize)]
pub(crate) struct Tables {
    table: Vec<[f32; 2]>,
    scale: [[f32; 2]; 2],
    offsets: [usize; 6],
}

/// Each node's measures, as the tree graph and the codes read them.
pub(crate) struct Nodes<'a> {
    pub(crate) templates: &'a [i64],
    pub(crate) anchors: &'a [i64],
    pub(crate) family: &'a [i64],
    pub(crate) value: &'a [i64],
    pub(crate) prefix: &'a [i64],
    pub(crate) section: &'a [i64],
    pub(crate) x0: &'a [f32],
    pub(crate) size: &'a [f32],
    pub(crate) looks: &'a [i64],
    pub(crate) pages: &'a [i64],
    pub(crate) roles: &'a [i64],
}

/// An outline anchor's levels (`docparse.ANCHORS`).
const ANCHORS: i64 = 10;

/// The pair scores of a tree: `score(i, j)`, node `i`'s parent and sibling scores for node `j` (j <= i).
pub(crate) struct Scores<'a> {
    nodes: Nodes<'a>,
    tables: &'a Tables,
    vectors: [&'a [f32]; 4],
    d: usize,
    /// Each node's latest earlier node of its marker family, where it has one.
    latest: Vec<Option<usize>>,
}

impl<'a> Scores<'a> {
    /// `vectors`: the parent query and key and the sibling query and key, `d` a node.
    pub(crate) fn new(nodes: Nodes<'a>, tables: &'a Tables, vectors: [&'a [f32]; 4], d: usize) -> Self {
        let mut last = std::collections::HashMap::new();
        let latest = nodes.family.iter().enumerate().map(|(i, &family)| {
            let earlier = (family > 0).then(|| last.get(&family).copied()).flatten();
            if family > 0 { last.insert(family, i); }
            earlier
        }).collect();
        Self { nodes, tables, vectors, d, latest }
    }

    /// The pair's six codes (`docparse.tree_codes`): the outline anchors' step, the grammar state, the geometry and
    /// style state, the marker prefixes' difference, the distance in reading order and a shared template.
    fn codes(&self, i: usize, j: usize) -> [usize; 6] {
        let n = &self.nodes;
        let (fi, fj, vi, vj) = (n.family[i], n.family[j], n.value[i], n.value[j]);
        let same_family = fi > 0 && fi == fj;
        let numbered = vi >= 0 && vj >= 0;
        let dv = vi - vj;
        let no_intervening = same_family && i > j && self.latest[i] == Some(j);
        let same_section = n.section[i] == n.section[j];
        let different_family = fi > 0 && fj > 0 && fi != fj;
        let grammar = if same_family && numbered {
            if dv <= 0 { 6 } else if dv > 1 { if no_intervening { 8 } else { 7 } } else if !no_intervening { 9 } else if same_section { 11 } else { 10 }
        } else if different_family {
            if vi != 1 { 5 } else if same_section { 3 } else { 4 }
        } else if fi == 0 && fj > 0 { 2 } else if fi > 0 && fj == 0 { 1 } else { 0 };
        let dx = n.x0[i] - n.x0[j];
        let geometry = if dx >= 0.015 { 16 } else if dx <= -0.015 { 32 } else { 0 }
            + 8 * usize::from(n.looks[i] == n.looks[j]) + 4 * usize::from(n.roles[i] == n.roles[j])
            + 2 * usize::from(n.pages[i] == n.pages[j]) + usize::from(same_section);
        let prefix = ((n.prefix[i] - n.prefix[j]).clamp(-4, 4) + 4) as usize;
        let distance = (((i.saturating_sub(j)) as f32 + 1.0).log2() as usize).min(15);
        let steps = (n.anchors[j] - n.anchors[i] + ANCHORS - 1).clamp(0, 2 * ANCHORS - 2) as usize;
        let same = usize::from(n.templates[i] == n.templates[j]);
        [steps, grammar, geometry, prefix, distance, same]
    }

    /// Node `i`'s parent and sibling scores for node `j`: the query-key products and the pair's biases.
    pub(crate) fn score(&self, i: usize, j: usize) -> (f32, f32) {
        let d = self.d;
        let dot = |q: &[f32], k: &[f32]| q[i * d..(i + 1) * d].iter().zip(&k[j * d..(j + 1) * d]).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt();
        let (mut parent, mut sibling) = (dot(self.vectors[0], self.vectors[1]), dot(self.vectors[2], self.vectors[3]));
        for (code, first) in self.codes(i, j).iter().zip(self.tables.offsets) {
            let row = self.tables.table[first + code];
            parent += row[0];
            sibling += row[1];
        }
        let n = &self.nodes;
        let dx = (n.x0[i] - n.x0[j]).clamp(-1.0, 1.0);
        let ds = (n.size[i] - n.size[j]).clamp(-2.0, 2.0);
        let scale = self.tables.scale;
        (parent + dx * scale[0][0] + ds * scale[1][0], sibling + dx * scale[0][1] + ds * scale[1][1])
    }
}
