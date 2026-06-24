use serde::{Deserialize, Serialize};

use crate::model::Tree;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Forest {
    pub forest_id: String,
    #[serde(default)]
    pub trees: Vec<Tree>,
}

impl Forest {
    pub fn new(forest_id: impl Into<String>) -> Self {
        Self {
            forest_id: forest_id.into(),
            trees: Vec::new(),
        }
    }

    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    pub fn add_tree(&mut self, tree: Tree) -> anyhow::Result<()> {
        if self.get_tree(&tree.node_id).is_some() {
            anyhow::bail!("duplicate tree id: {}", tree.node_id);
        }
        self.trees.push(tree);
        Ok(())
    }

    pub fn get_tree(&self, tree_id: &str) -> Option<&Tree> {
        self.trees.iter().find(|tree| tree.node_id == tree_id)
    }
}
