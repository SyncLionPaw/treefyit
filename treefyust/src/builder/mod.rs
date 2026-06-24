use std::fs;
use std::path::Path;

use anyhow::Result;

use crate::config::BuilderSettings;
use crate::model::{LeafType, NodeContent, Tree};

#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    pub summarize: bool,
}

#[derive(Debug, Clone)]
pub struct Section {
    pub title: String,
    pub level: usize,
    pub text: String,
    pub summary: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RuleBasedSectionRefiner {
    pub split_threshold: usize,
    pub max_parts: usize,
}

impl Default for RuleBasedSectionRefiner {
    fn default() -> Self {
        let settings = BuilderSettings::default();
        Self {
            split_threshold: settings.refine_split_threshold,
            max_parts: settings.refine_max_parts,
        }
    }
}

impl RuleBasedSectionRefiner {
    pub fn new(split_threshold: Option<usize>, max_parts: Option<usize>) -> Self {
        let settings = BuilderSettings::default();
        Self {
            split_threshold: split_threshold.unwrap_or(settings.refine_split_threshold),
            max_parts: max_parts.unwrap_or(settings.refine_max_parts),
        }
    }

    pub fn refine(&self, sections: Vec<Section>) -> Vec<Section> {
        let mut refined = Vec::new();
        for section in sections {
            if section.text.chars().count() <= self.split_threshold {
                refined.push(section);
                continue;
            }
            refined.extend(self.split_long_section(section));
        }
        refined
    }

    fn split_long_section(&self, section: Section) -> Vec<Section> {
        let chunks = split_text(&section.text, self.max_parts);
        if chunks.len() <= 1 {
            return vec![section];
        }

        let summary = section
            .summary
            .clone()
            .or_else(|| summarize_text(&section.text));
        let mut result = vec![Section {
            title: section.title.clone(),
            level: section.level,
            text: String::new(),
            summary,
        }];

        for (index, chunk) in chunks.into_iter().enumerate() {
            result.push(Section {
                title: format!("{} Part {}", section.title, index + 1),
                level: section.level + 1,
                text: chunk,
                summary: None,
            });
        }
        result
    }
}

pub fn build_tree_from_text(text: &str, filename: &str, options: BuildOptions) -> Tree {
    build_tree_from_text_with_refiner(text, filename, options, RuleBasedSectionRefiner::default())
}

pub fn build_tree_from_text_with_refiner(
    text: &str,
    filename: &str,
    options: BuildOptions,
    refiner: RuleBasedSectionRefiner,
) -> Tree {
    let sections = parse_sections(text);
    let sections = refiner.refine(sections);
    build_tree_from_sections(sections, filename, options)
}

pub fn build_tree_from_file(path: impl AsRef<Path>, options: BuildOptions) -> Result<Tree> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("document.md");
    Ok(build_tree_from_text(&text, filename, options))
}

pub fn parse_sections(text: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut current_title: Option<String> = None;
    let mut current_level = 1;
    let mut current_lines: Vec<String> = Vec::new();

    for line in text.lines() {
        if let Some((level, title)) = parse_heading(line) {
            if let Some(title) = current_title.take() {
                sections.push(Section {
                    title,
                    level: current_level,
                    text: current_lines.join("\n").trim().to_string(),
                    summary: None,
                });
                current_lines.clear();
            }
            current_title = Some(title);
            current_level = level;
            continue;
        }

        if current_title.is_some() {
            current_lines.push(line.to_string());
        }
    }

    if let Some(title) = current_title.take() {
        sections.push(Section {
            title,
            level: current_level,
            text: current_lines.join("\n").trim().to_string(),
            summary: None,
        });
    }

    if sections.is_empty() {
        let stripped = text.trim();
        if !stripped.is_empty() {
            sections.push(Section {
                title: "Document".to_string(),
                level: 1,
                text: stripped.to_string(),
                summary: None,
            });
        }
    }

    sections
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|ch| *ch == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = trimmed[level..].trim();
    if rest.is_empty() {
        return None;
    }
    Some((level, rest.to_string()))
}

pub fn build_tree_from_sections(
    sections: Vec<Section>,
    root_title: &str,
    options: BuildOptions,
) -> Tree {
    let mut root = Tree::new("document", root_title);
    let mut stack: Vec<(usize, Vec<usize>)> = Vec::new();

    for section in sections {
        while stack
            .last()
            .is_some_and(|(level, _)| *level >= section.level)
        {
            stack.pop();
        }
        let parent_path = stack
            .last()
            .map(|(_, path)| path.clone())
            .unwrap_or_default();
        let next_index = get_tree_by_path(&root, &parent_path)
            .map(|tree| tree.children.len())
            .unwrap_or(0);
        let mut child_path = parent_path.clone();
        child_path.push(next_index);

        let node_id = format!("node-{}", count_nodes(&root));
        let mut node = Tree::new(node_id, section.title);
        if !section.text.trim().is_empty() {
            node.content = Some(NodeContent::Text { text: section.text });
            node.leaf_type = Some(LeafType::Text);
        }
        node.summary = section.summary;
        if options.summarize
            && node.summary.is_none()
            && let Some(NodeContent::Text { text }) = &node.content
        {
            node.summary = summarize_text(text);
        }

        if let Some(parent) = get_tree_by_path_mut(&mut root, &parent_path) {
            parent.children.push(node);
        }
        stack.push((section.level, child_path));
    }

    fill_tree_stats(&mut root, 0);
    root
}

fn get_tree_by_path<'a>(tree: &'a Tree, path: &[usize]) -> Option<&'a Tree> {
    let mut current = tree;
    for index in path {
        current = current.children.get(*index)?;
    }
    Some(current)
}

fn get_tree_by_path_mut<'a>(tree: &'a mut Tree, path: &[usize]) -> Option<&'a mut Tree> {
    let mut current = tree;
    for index in path {
        current = current.children.get_mut(*index)?;
    }
    Some(current)
}

fn count_nodes(tree: &Tree) -> usize {
    1 + tree.children.iter().map(count_nodes).sum::<usize>()
}

fn fill_tree_stats(tree: &mut Tree, depth: usize) -> (usize, usize) {
    tree.depth = Some(depth);
    if tree.children.is_empty() {
        tree.subtree_size = Some(1);
        tree.leaf_count = Some(1);
        return (1, 1);
    }

    let mut subtree_size = 1;
    let mut leaf_count = 0;
    for child in &mut tree.children {
        let (child_size, child_leaves) = fill_tree_stats(child, depth + 1);
        subtree_size += child_size;
        leaf_count += child_leaves;
    }
    tree.subtree_size = Some(subtree_size);
    tree.leaf_count = Some(leaf_count);
    (subtree_size, leaf_count)
}

fn split_text(text: &str, max_parts: usize) -> Vec<String> {
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if paragraphs.len() >= 2 {
        return paragraphs
            .chunks((paragraphs.len().div_ceil(max_parts)).max(1))
            .map(|chunk| chunk.join("\n\n"))
            .collect();
    }

    let chars: Vec<char> = text.chars().collect();
    let chunk_size = chars.len().div_ceil(max_parts).max(1);
    chars
        .chunks(chunk_size)
        .map(|chunk| chunk.iter().collect::<String>().trim().to_string())
        .filter(|part| !part.is_empty())
        .collect()
}

fn summarize_text(text: &str) -> Option<String> {
    let stripped = text.trim();
    if stripped.is_empty() {
        return None;
    }
    let summary: String = stripped.chars().take(120).collect();
    Some(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_tree_from_markdown_text() {
        let tree = build_tree_from_text(
            "# Intro\n\nHello world.\n\n## Detail\n\nMore text.",
            "sample.md",
            BuildOptions::default(),
        );

        assert_eq!(tree.title, "sample.md");
        assert_eq!(tree.subtree_size, Some(3));
        assert_eq!(tree.children[0].title, "Intro");
        assert_eq!(tree.children[0].children[0].title, "Detail");
    }

    #[test]
    fn split_long_section_keeps_parent_as_summary_node() {
        let refiner = RuleBasedSectionRefiner::new(Some(10), Some(2));
        let sections = refiner.refine(vec![Section {
            title: "Long".to_string(),
            level: 1,
            text: "alpha beta gamma delta epsilon".to_string(),
            summary: None,
        }]);

        assert!(sections.len() > 1);
        assert_eq!(sections[0].title, "Long");
        assert!(sections[0].text.is_empty());
        assert!(sections[0].summary.is_some());
        assert_eq!(sections[1].level, 2);
    }
}
