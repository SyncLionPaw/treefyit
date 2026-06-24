use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LeafType {
    Text,
    Image,
    Table,
    Link,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextContent {
    #[serde(default = "text_kind")]
    pub kind: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrlContent {
    #[serde(default = "url_kind")]
    pub kind: String,
    pub url: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceContent {
    #[serde(default = "resource_kind")]
    pub kind: String,
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum NodeContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "url")]
    Url { url: String },
    #[serde(rename = "resource")]
    Resource {
        uri: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub node_id: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<NodeContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leaf_type: Option<LeafType>,
    #[serde(default)]
    pub children: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tree {
    pub node_id: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<NodeContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leaf_type: Option<LeafType>,
    #[serde(default)]
    pub children: Vec<Tree>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtree_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leaf_count: Option<usize>,
}

impl Tree {
    pub fn new(node_id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            title: title.into(),
            content: None,
            summary: None,
            leaf_type: None,
            children: Vec::new(),
            depth: None,
            subtree_size: None,
            leaf_count: None,
        }
    }

    pub fn with_text(
        node_id: impl Into<String>,
        title: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        let mut tree = Self::new(node_id, title);
        tree.content = Some(NodeContent::Text { text: text.into() });
        tree.leaf_type = Some(LeafType::Text);
        tree
    }

    pub fn add_child(&mut self, child: Tree) {
        self.children.push(child);
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn child_count(&self) -> usize {
        self.children.len()
    }
}

fn text_kind() -> String {
    "text".to_string()
}

fn url_kind() -> String {
    "url".to_string()
}

fn resource_kind() -> String {
    "resource".to_string()
}
