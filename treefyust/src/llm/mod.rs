use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{LlmSettings, get_settings};
use crate::model::{NodeContent, Tree};

pub const SUMMARY_SYSTEM_PROMPT: &str = r#"你是一个摘要专家。
请根据给定内容生成准确、简洁、忠实原文的中文摘要。
不要编造信息，不要输出项目符号，不要输出解释。"#;

const SUMMARY_USER_PROMPT_TEMPLATE: &str = r#"请为当前节点生成摘要。

要求：
1. 保留主题、结论和关键信息
2. 输出 2-4 句
3. 如果正文较少，可以参考子节点摘要
4. 不要重复标题原文
5. 不要使用“本节主要讲了”这类空话

当前节点标题：
{title}

当前节点正文：
{content}

子节点摘要：
{children_summaries}
"#;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub fn configured_model() -> Result<String> {
    Ok(get_settings()?.llm.model)
}

pub fn build_messages(prompt: &str, system: Option<&str>) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system) = system
        && !system.trim().is_empty()
    {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system.to_string(),
        });
    }
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: prompt.to_string(),
    });
    messages
}

pub fn format_child_summaries(child_summaries: &[String]) -> String {
    let lines: Vec<&str> = child_summaries
        .iter()
        .map(|summary| summary.trim())
        .filter(|summary| !summary.is_empty())
        .collect();
    if lines.is_empty() {
        return "无".to_string();
    }
    lines
        .into_iter()
        .map(|summary| format!("- {summary}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn build_summary_prompt(title: &str, content: &str, child_summaries: &[String]) -> String {
    SUMMARY_USER_PROMPT_TEMPLATE
        .replace("{title}", non_empty_or(title, "Untitled"))
        .replace("{content}", non_empty_or(content, "无"))
        .replace(
            "{children_summaries}",
            &format_child_summaries(child_summaries),
        )
}

pub async fn summarize_text(
    title: &str,
    content: &str,
    child_summaries: &[String],
) -> Result<String> {
    let prompt = build_summary_prompt(title, content, child_summaries);
    let text = complete(&prompt, Some(SUMMARY_SYSTEM_PROMPT), None, None, None).await?;
    Ok(text.trim().to_string())
}

pub async fn summarize_tree(tree: &mut Tree) -> Result<()> {
    let paths = collect_node_paths_postorder(tree);
    for path in paths {
        let Some((title, content, child_summaries)) = summary_input(tree, &path) else {
            continue;
        };
        let summary = summarize_text(&title, &content, &child_summaries).await?;
        let summary = summary.trim();
        if summary.is_empty() {
            continue;
        }
        if let Some(node) = tree_by_path_mut(tree, &path) {
            node.summary = Some(summary.to_string());
        }
    }
    Ok(())
}

pub fn summarize_tree_with_generator<F>(tree: &mut Tree, mut generate_summary: F) -> Result<()>
where
    F: FnMut(&str, &str, &[String]) -> Result<String>,
{
    let paths = collect_node_paths_postorder(tree);
    for path in paths {
        let Some((title, content, child_summaries)) = summary_input(tree, &path) else {
            continue;
        };
        let summary = generate_summary(&title, &content, &child_summaries)?;
        let summary = summary.trim();
        if summary.is_empty() {
            continue;
        }
        if let Some(node) = tree_by_path_mut(tree, &path) {
            node.summary = Some(summary.to_string());
        }
    }
    Ok(())
}

pub async fn complete(
    prompt: &str,
    system: Option<&str>,
    model: Option<&str>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
) -> Result<String> {
    let mut settings = get_settings()?.llm;
    if let Some(model) = model {
        settings.model = model.to_string();
    }
    if let Some(temperature) = temperature {
        settings.temperature = temperature;
    }
    if let Some(max_tokens) = max_tokens {
        settings.max_tokens = Some(max_tokens);
    }
    complete_with_settings(prompt, system, &settings).await
}

pub async fn complete_with_settings(
    prompt: &str,
    system: Option<&str>,
    settings: &LlmSettings,
) -> Result<String> {
    let base_url = settings
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .context("llm.base_url is required")?;
    let client = Client::new();
    let model = normalize_model_name(&settings.model);
    let messages = build_messages(prompt, system);

    if is_ollama_target(&settings.model, base_url) {
        let mut options = json!({
            "temperature": settings.temperature,
        });
        if let Some(max_tokens) = settings.max_tokens {
            options["num_predict"] = json!(max_tokens);
        }
        let response = client
            .post(format!("{}/api/chat", base_url.trim_end_matches('/')))
            .json(&json!({
                "model": model,
                "messages": messages,
                "stream": false,
                "options": options,
            }))
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;
        return extract_response_text(&response);
    }

    let mut request = client
        .post(format!(
            "{}/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .json(&json!({
            "model": model,
            "messages": messages,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        }));
    if let Some(api_key) = settings
        .api_key
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        request = request.bearer_auth(api_key);
    }
    let response = request
        .send()
        .await?
        .error_for_status()?
        .json::<Value>()
        .await?;
    extract_response_text(&response)
}

pub fn extract_response_text(response: &Value) -> Result<String> {
    if let Some(text) = response
        .pointer("/message/content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        return Ok(text.to_string());
    }

    if let Some(text) = response
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        return Ok(text.to_string());
    }

    if let Some(items) = response
        .pointer("/choices/0/message/content")
        .and_then(Value::as_array)
    {
        let texts: Vec<&str> = items
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect();
        if !texts.is_empty() {
            return Ok(texts.join("\n"));
        }
    }

    if let Some(text) = response
        .get("response")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        return Ok(text.to_string());
    }

    bail!("LLM response did not contain text content")
}

pub fn count_tokens(text: &str) -> usize {
    crate::query::tokenize(text).len()
}

fn normalize_model_name(model: &str) -> &str {
    model
        .strip_prefix("litellm/")
        .unwrap_or(model)
        .strip_prefix("ollama/")
        .unwrap_or_else(|| model.strip_prefix("litellm/").unwrap_or(model))
}

fn is_ollama_target(model: &str, base_url: &str) -> bool {
    model.starts_with("ollama/") || base_url.contains(":11434")
}

fn non_empty_or<'a>(value: &'a str, fallback: &'a str) -> &'a str {
    let value = value.trim();
    if value.is_empty() {
        return fallback;
    }
    value
}

fn collect_node_paths_postorder(tree: &Tree) -> Vec<Vec<usize>> {
    let mut paths = Vec::new();
    append_node_paths_postorder(tree, Vec::new(), &mut paths);
    paths
}

fn append_node_paths_postorder(tree: &Tree, path: Vec<usize>, paths: &mut Vec<Vec<usize>>) {
    for (index, child) in tree.children.iter().enumerate() {
        let mut child_path = path.clone();
        child_path.push(index);
        append_node_paths_postorder(child, child_path, paths);
    }
    paths.push(path);
}

fn summary_input(tree: &Tree, path: &[usize]) -> Option<(String, String, Vec<String>)> {
    let node = tree_by_path(tree, path)?;
    let content = content_text(node.content.as_ref());
    let child_summaries = node
        .children
        .iter()
        .filter_map(|child| child.summary.clone())
        .collect::<Vec<_>>();
    if content.trim().is_empty() && child_summaries.is_empty() {
        return None;
    }
    Some((node.title.clone(), content, child_summaries))
}

fn tree_by_path<'a>(tree: &'a Tree, path: &[usize]) -> Option<&'a Tree> {
    let mut current = tree;
    for index in path {
        current = current.children.get(*index)?;
    }
    Some(current)
}

fn tree_by_path_mut<'a>(tree: &'a mut Tree, path: &[usize]) -> Option<&'a mut Tree> {
    let mut current = tree;
    for index in path {
        current = current.children.get_mut(*index)?;
    }
    Some(current)
}

fn content_text(content: Option<&NodeContent>) -> String {
    match content {
        Some(NodeContent::Text { text }) => text.clone(),
        Some(NodeContent::Url { url }) => url.clone(),
        Some(NodeContent::Resource { uri, .. }) => uri.clone(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn builds_messages_with_optional_system_prompt() {
        let messages = build_messages("hello", Some("system"));

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].content, "hello");
    }

    #[test]
    fn builds_summary_prompt_with_child_summaries() {
        let prompt = build_summary_prompt(
            "Intro",
            "Detailed content.",
            &["child one".to_string(), "child two".to_string()],
        );

        assert!(prompt.contains("Intro"));
        assert!(prompt.contains("Detailed content."));
        assert!(prompt.contains("- child one"));
        assert!(prompt.contains("- child two"));
    }

    #[test]
    fn extracts_ollama_and_openai_response_text() {
        assert_eq!(
            extract_response_text(&json!({"message": {"content": "ollama text"}})).unwrap(),
            "ollama text"
        );
        assert_eq!(
            extract_response_text(&json!({"choices": [{"message": {"content": "openai text"}}]}))
                .unwrap(),
            "openai text"
        );
        assert_eq!(
            extract_response_text(&json!({
                "choices": [{
                    "message": {
                        "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]
                    }
                }]
            }))
            .unwrap(),
            "part one\npart two"
        );
    }

    #[test]
    fn normalizes_litellm_and_ollama_model_prefixes() {
        assert_eq!(
            normalize_model_name("ollama/gemma4:latest"),
            "gemma4:latest"
        );
        assert_eq!(normalize_model_name("litellm/gpt-4o"), "gpt-4o");
    }

    #[test]
    fn summarizes_tree_bottom_up_with_generator() {
        let mut tree = Tree::new("doc", "Document");
        let mut parent = Tree::new("node-1", "Parent");
        parent.add_child(Tree::with_text("node-2", "Child", "child body"));
        tree.add_child(parent);

        summarize_tree_with_generator(&mut tree, |title, content, child_summaries| {
            Ok(format!(
                "{title}|{}|{}",
                content.trim(),
                child_summaries.join(",")
            ))
        })
        .unwrap();

        let parent = &tree.children[0];
        let child = &parent.children[0];
        assert_eq!(child.summary.as_deref(), Some("Child|child body|"));
        assert_eq!(parent.summary.as_deref(), Some("Parent||Child|child body|"));
        assert_eq!(
            tree.summary.as_deref(),
            Some("Document||Parent||Child|child body|")
        );
    }
}
