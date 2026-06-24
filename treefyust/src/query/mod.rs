use std::collections::{HashMap, HashSet};

use jieba_rs::Jieba;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

use crate::model::{Forest, NodeContent, Tree};

static JIEBA: Lazy<Jieba> = Lazy::new(Jieba::new);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeQueryHit {
    pub tree_id: String,
    pub tree_title: String,
    pub score: f64,
    pub node_count: usize,
    pub root_titles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeQueryHit {
    pub tree_id: String,
    pub tree_title: String,
    pub node_id: String,
    pub path: String,
    pub title: String,
    pub score: f64,
    pub summary: Option<String>,
    pub snippet: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Posting {
    pub term_frequency: usize,
    pub node_id: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeDocumentStats {
    pub node_id: String,
    pub path: String,
    pub title: String,
    pub summary: Option<String>,
    pub text: String,
    pub title_term_frequency: HashMap<String, usize>,
    pub summary_term_frequency: HashMap<String, usize>,
    pub content_term_frequency: HashMap<String, usize>,
    pub term_frequency: HashMap<String, usize>,
    pub document_length: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CorpusStats {
    pub document_count: usize,
    pub average_document_length: f64,
    pub document_frequency: HashMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeIndex {
    pub tree_id: String,
    pub tree_title: String,
    pub node_documents: Vec<NodeDocumentStats>,
    pub postings: HashMap<String, Vec<Posting>>,
    pub corpus: CorpusStats,
    pub tree_term_frequency: HashMap<String, usize>,
    pub tree_document_length: usize,
}

#[derive(Debug, Clone)]
struct NodeRow {
    tree_id: String,
    tree_title: String,
    node_id: String,
    path: String,
    title: String,
    summary: Option<String>,
    text: String,
    search_text: String,
}

#[derive(Debug, Clone)]
struct TreeRow {
    tree_id: String,
    tree_title: String,
    node_count: usize,
    root_titles: Vec<String>,
    search_text: String,
}

pub struct InMemoryForestQuery {
    forest: Forest,
    tree_rows: Vec<TreeRow>,
    node_rows: Vec<NodeRow>,
}

impl InMemoryForestQuery {
    pub fn new(forest: Forest) -> Self {
        let mut query = Self {
            forest,
            tree_rows: Vec::new(),
            node_rows: Vec::new(),
        };
        query.rebuild();
        query
    }

    pub fn rebuild(&mut self) {
        self.tree_rows.clear();
        self.node_rows.clear();
        for tree in &self.forest.trees {
            self.tree_rows.push(build_tree_row(tree));
            self.node_rows.extend(flatten_tree_nodes(tree));
        }
    }

    pub fn find_trees(&self, query: &str, limit: usize) -> Vec<TreeQueryHit> {
        let query = query.trim();
        if query.is_empty() {
            return Vec::new();
        }
        let corpus: Vec<Vec<String>> = self
            .tree_rows
            .iter()
            .map(|row| tokenize(&row.search_text))
            .collect();
        let scores = bm25_scores(&corpus, &tokenize(query));
        let mut ranked: Vec<(&TreeRow, f64)> = self.tree_rows.iter().zip(scores).collect();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        ranked
            .into_iter()
            .filter(|(_, score)| *score > 0.0)
            .take(limit)
            .map(|(row, score)| TreeQueryHit {
                tree_id: row.tree_id.clone(),
                tree_title: row.tree_title.clone(),
                score: round_score(score),
                node_count: row.node_count,
                root_titles: row.root_titles.iter().take(8).cloned().collect(),
            })
            .collect()
    }

    pub fn find_nodes(&self, query: &str, limit: usize) -> Vec<NodeQueryHit> {
        let query = query.trim();
        if query.is_empty() {
            return Vec::new();
        }
        let corpus: Vec<Vec<String>> = self
            .node_rows
            .iter()
            .map(|row| tokenize(&row.search_text))
            .collect();
        let scores = bm25_scores(&corpus, &tokenize(query));
        let mut ranked: Vec<(&NodeRow, f64)> = self.node_rows.iter().zip(scores).collect();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        ranked
            .into_iter()
            .filter(|(_, score)| *score > 0.0)
            .take(limit)
            .map(|(row, score)| NodeQueryHit {
                tree_id: row.tree_id.clone(),
                tree_title: row.tree_title.clone(),
                node_id: row.node_id.clone(),
                path: row.path.clone(),
                title: row.title.clone(),
                score: round_score(score),
                summary: row.summary.clone(),
                snippet: snippet(row),
            })
            .collect()
    }
}

pub fn build_tree_index(tree: &Tree) -> TreeIndex {
    let rows = flatten_tree_nodes(tree);
    let mut node_documents = Vec::new();
    let mut postings: HashMap<String, Vec<Posting>> = HashMap::new();
    let mut document_frequency: HashMap<String, usize> = HashMap::new();
    let mut tree_term_frequency: HashMap<String, usize> = HashMap::new();
    let mut total_document_length = 0;

    for row in rows {
        let title_term_frequency = count_terms(&row.title);
        let summary_term_frequency = count_terms(row.summary.as_deref().unwrap_or(""));
        let content_term_frequency = count_terms(&row.text);
        let term_frequency = merge_term_frequencies(&[
            title_term_frequency.clone(),
            summary_term_frequency.clone(),
            content_term_frequency.clone(),
        ]);
        let document_length = term_frequency.values().sum();
        total_document_length += document_length;

        for (term, frequency) in &term_frequency {
            *tree_term_frequency.entry(term.clone()).or_insert(0) += frequency;
            postings.entry(term.clone()).or_default().push(Posting {
                term_frequency: *frequency,
                node_id: row.node_id.clone(),
                path: row.path.clone(),
            });
        }

        for term in term_frequency.keys().collect::<HashSet<_>>() {
            *document_frequency.entry(term.clone()).or_insert(0) += 1;
        }

        node_documents.push(NodeDocumentStats {
            node_id: row.node_id,
            path: row.path,
            title: row.title,
            summary: row.summary,
            text: row.text,
            title_term_frequency,
            summary_term_frequency,
            content_term_frequency,
            term_frequency,
            document_length,
        });
    }

    let document_count = node_documents.len();
    let average_document_length = if document_count == 0 {
        0.0
    } else {
        total_document_length as f64 / document_count as f64
    };
    let tree_document_length = tree_term_frequency.values().sum();

    TreeIndex {
        tree_id: tree.node_id.clone(),
        tree_title: tree.title.clone(),
        node_documents,
        postings,
        corpus: CorpusStats {
            document_count,
            average_document_length,
            document_frequency,
        },
        tree_term_frequency,
        tree_document_length,
    }
}

pub fn score_nodes_bm25(index: &TreeIndex, query: &str, limit: usize) -> Vec<NodeQueryHit> {
    let query_terms = count_terms(query);
    if query_terms.is_empty() {
        return Vec::new();
    }

    let documents: HashMap<&str, &NodeDocumentStats> = index
        .node_documents
        .iter()
        .map(|document| (document.node_id.as_str(), document))
        .collect();
    let mut scores: HashMap<String, f64> = HashMap::new();
    let average_document_length = index.corpus.average_document_length.max(1.0);
    let document_count = index.corpus.document_count as f64;
    let k1 = 1.5;
    let b = 0.75;

    for (term, query_frequency) in query_terms {
        let Some(postings) = index.postings.get(&term) else {
            continue;
        };
        let document_frequency = *index.corpus.document_frequency.get(&term).unwrap_or(&0) as f64;
        if document_frequency <= 0.0 {
            continue;
        }
        let idf =
            (1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)).ln();
        for posting in postings {
            let Some(document) = documents.get(posting.node_id.as_str()) else {
                continue;
            };
            let document_length = document.document_length.max(1) as f64;
            let term_frequency = posting.term_frequency as f64;
            let numerator = term_frequency * (k1 + 1.0);
            let denominator =
                term_frequency + k1 * (1.0 - b + b * document_length / average_document_length);
            let contribution = query_frequency as f64 * idf * numerator / denominator;
            *scores.entry(posting.node_id.clone()).or_insert(0.0) += contribution;
        }
    }

    let mut ranked: Vec<(&NodeDocumentStats, f64)> = scores
        .iter()
        .filter_map(|(node_id, score)| documents.get(node_id.as_str()).map(|doc| (*doc, *score)))
        .filter(|(_, score)| *score > 0.0)
        .collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    ranked
        .into_iter()
        .take(limit)
        .map(|(document, score)| NodeQueryHit {
            tree_id: index.tree_id.clone(),
            tree_title: index.tree_title.clone(),
            node_id: document.node_id.clone(),
            path: document.path.clone(),
            title: document.title.clone(),
            score: round_score(score),
            summary: document.summary.clone(),
            snippet: snippet_from_parts(
                document.summary.as_deref(),
                Some(document.text.as_str()),
                Some(document.title.as_str()),
            ),
        })
        .collect()
}

fn build_tree_row(tree: &Tree) -> TreeRow {
    let root_titles = tree
        .children
        .iter()
        .map(|child| child.title.clone())
        .collect();
    let node_rows = flatten_tree_nodes(tree);
    let search_text = node_rows
        .iter()
        .flat_map(|row| {
            [
                row.title.as_str(),
                row.summary.as_deref().unwrap_or(""),
                row.text.as_str(),
            ]
        })
        .chain([tree.node_id.as_str(), tree.title.as_str()])
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    TreeRow {
        tree_id: tree.node_id.clone(),
        tree_title: tree.title.clone(),
        node_count: node_rows.len(),
        root_titles,
        search_text,
    }
}

fn flatten_tree_nodes(tree: &Tree) -> Vec<NodeRow> {
    let mut rows = Vec::new();
    append_node_rows(tree, &mut rows, &tree.node_id, &tree.title, "root");
    rows
}

fn append_node_rows(
    node: &Tree,
    rows: &mut Vec<NodeRow>,
    tree_id: &str,
    tree_title: &str,
    path: &str,
) {
    let text = content_to_search_text(node.content.as_ref());
    let search_text = [&node.title, node.summary.as_deref().unwrap_or(""), &text]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    rows.push(NodeRow {
        tree_id: tree_id.to_string(),
        tree_title: tree_title.to_string(),
        node_id: node.node_id.clone(),
        path: path.to_string(),
        title: node.title.clone(),
        summary: node.summary.clone(),
        text,
        search_text,
    });

    for (index, child) in node.children.iter().enumerate() {
        let child_path = if path == "root" {
            index.to_string()
        } else {
            format!("{path}.{index}")
        };
        append_node_rows(child, rows, tree_id, tree_title, &child_path);
    }
}

pub fn content_to_search_text(content: Option<&NodeContent>) -> String {
    match content {
        Some(NodeContent::Text { text }) => text.clone(),
        Some(NodeContent::Url { url }) => url.clone(),
        Some(NodeContent::Resource { uri, .. }) => uri.clone(),
        None => String::new(),
    }
}

pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_is_ascii = None;

    for ch in text.to_lowercase().chars() {
        let kind = if ch.is_ascii_alphanumeric() {
            Some(true)
        } else if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            Some(false)
        } else {
            None
        };

        if kind.is_none() {
            flush_token_chunk(&mut tokens, &current, current_is_ascii);
            current.clear();
            current_is_ascii = None;
            continue;
        }

        if current_is_ascii != kind && !current.is_empty() {
            flush_token_chunk(&mut tokens, &current, current_is_ascii);
            current.clear();
        }
        current_is_ascii = kind;
        current.push(ch);
    }
    flush_token_chunk(&mut tokens, &current, current_is_ascii);

    let stripped = text.trim().to_lowercase();
    if tokens.is_empty() && !stripped.is_empty() {
        tokens.push(stripped);
    }
    tokens
}

fn flush_token_chunk(tokens: &mut Vec<String>, chunk: &str, is_ascii: Option<bool>) {
    if chunk.is_empty() {
        return;
    }
    if is_ascii == Some(true) {
        tokens.push(chunk.to_string());
        return;
    }
    for token in JIEBA.cut(chunk, false) {
        let token = token.trim();
        if !token.is_empty() {
            tokens.push(token.to_string());
        }
    }
}

fn count_terms(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for token in tokenize(text) {
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
}

fn merge_term_frequencies(parts: &[HashMap<String, usize>]) -> HashMap<String, usize> {
    let mut merged = HashMap::new();
    for part in parts {
        for (term, frequency) in part {
            *merged.entry(term.clone()).or_insert(0) += frequency;
        }
    }
    merged
}

fn bm25_scores(corpus: &[Vec<String>], query_tokens: &[String]) -> Vec<f64> {
    if corpus.is_empty() || query_tokens.is_empty() {
        return vec![0.0; corpus.len()];
    }

    let mut doc_term_counts = Vec::new();
    let mut document_frequency: HashMap<String, usize> = HashMap::new();
    let mut total_length = 0;
    for document in corpus {
        let mut counts = HashMap::new();
        let mut seen = HashSet::new();
        total_length += document.len();
        for token in document {
            *counts.entry(token.clone()).or_insert(0) += 1;
            seen.insert(token.clone());
        }
        for token in seen {
            *document_frequency.entry(token).or_insert(0) += 1;
        }
        doc_term_counts.push(counts);
    }

    let avgdl = (total_length as f64 / corpus.len() as f64).max(1.0);
    let n = corpus.len() as f64;
    let k1 = 1.5;
    let b = 0.75;

    corpus
        .iter()
        .zip(doc_term_counts.iter())
        .map(|(document, counts)| {
            let dl = document.len().max(1) as f64;
            query_tokens.iter().fold(0.0, |score, term| {
                let tf = *counts.get(term).unwrap_or(&0) as f64;
                if tf <= 0.0 {
                    return score;
                }
                let df = *document_frequency.get(term).unwrap_or(&0) as f64;
                let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                let numerator = tf * (k1 + 1.0);
                let denominator = tf + k1 * (1.0 - b + b * dl / avgdl);
                score + idf * numerator / denominator
            })
        })
        .collect()
}

fn snippet(row: &NodeRow) -> String {
    snippet_from_parts(
        row.summary.as_deref(),
        Some(row.text.as_str()),
        Some(row.title.as_str()),
    )
}

fn snippet_from_parts(summary: Option<&str>, text: Option<&str>, title: Option<&str>) -> String {
    for value in [summary, text, title].into_iter().flatten() {
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        return value.chars().take(160).collect();
    }
    String::new()
}

fn round_score(score: f64) -> f64 {
    (score * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use crate::model::{Forest, NodeContent, Tree};

    use super::*;

    fn sample_forest() -> Forest {
        let mut retrieval = Tree::with_text(
            "doc-retrieval",
            "Retrieval Paper",
            "A paper about document understanding.",
        );
        retrieval.add_child(Tree::with_text(
            "doc-retrieval-0",
            "Introduction",
            "General introduction.",
        ));
        let mut dense = Tree::with_text(
            "doc-retrieval-1",
            "Dense Retrieval",
            "This section covers retrieval, reranking, and recall.",
        );
        dense.summary = Some("retrieval section".to_string());
        retrieval.add_child(dense);

        let vision = Tree {
            node_id: "doc-vision".to_string(),
            title: "Vision Notes".to_string(),
            content: Some(NodeContent::Text {
                text: "Image parsing and visual understanding.".to_string(),
            }),
            summary: None,
            leaf_type: None,
            children: vec![Tree::with_text(
                "doc-vision-0",
                "Tables",
                "How to parse tables from html and pdf.",
            )],
            depth: None,
            subtree_size: None,
            leaf_count: None,
        };

        Forest {
            forest_id: "forest-demo".to_string(),
            trees: vec![retrieval, vision],
        }
    }

    #[test]
    fn tree_index_collects_terms() {
        let tree = sample_forest().trees.remove(0);
        let index = build_tree_index(&tree);

        assert_eq!(index.tree_id, "doc-retrieval");
        assert_eq!(index.corpus.document_count, 3);
        assert!(index.tree_term_frequency["retrieval"] >= 2);
    }

    #[test]
    fn bm25_returns_best_node() {
        let tree = sample_forest().trees.remove(0);
        let index = build_tree_index(&tree);
        let result = score_nodes_bm25(&index, "reranking recall", 8);

        assert_eq!(result[0].node_id, "doc-retrieval-1");
        assert!(result[0].score > 0.0);
    }

    #[test]
    fn forest_query_finds_nodes_and_trees() {
        let query = InMemoryForestQuery::new(sample_forest());

        assert_eq!(
            query.find_trees("reranking recall", 5)[0].tree_id,
            "doc-retrieval"
        );
        assert_eq!(query.find_nodes("tables", 5)[0].tree_id, "doc-vision");
    }

    #[test]
    fn tokenizes_chinese_words() {
        let tokens = tokenize("这里讨论摘要生成和结构化处理。");
        assert!(tokens.iter().any(|token| token == "摘要" || token == "摘"));
    }
}
