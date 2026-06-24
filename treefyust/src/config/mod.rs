use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use toml::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmSettings {
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub temperature: f32,
    pub max_tokens: Option<usize>,
}

impl Default for LlmSettings {
    fn default() -> Self {
        Self {
            model: "ollama/gemma4:latest".to_string(),
            api_key: None,
            base_url: Some("http://127.0.0.1:11434".to_string()),
            temperature: 0.0,
            max_tokens: Some(2048),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MinerUSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BuilderSettings {
    pub refine_split_threshold: usize,
    pub refine_max_parts: usize,
}

impl Default for BuilderSettings {
    fn default() -> Self {
        Self {
            refine_split_threshold: 400,
            refine_max_parts: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StoreSettings {
    pub data_dir: PathBuf,
}

impl Default for StoreSettings {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from(".treefyust-store"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppSettings {
    pub llm: LlmSettings,
    pub mineru: MinerUSettings,
    pub builder: BuilderSettings,
    pub store: StoreSettings,
}

pub fn config_paths() -> Vec<PathBuf> {
    vec![
        PathBuf::from("config/settings.toml"),
        PathBuf::from("config/settings.local.toml"),
        PathBuf::from("treefyust/config/settings.toml"),
        PathBuf::from("treefyust/config/settings.local.toml"),
    ]
}

pub fn get_settings() -> Result<AppSettings> {
    build_settings_from_paths(&config_paths())
}

pub fn build_settings_from_paths(paths: &[PathBuf]) -> Result<AppSettings> {
    let mut merged = Value::Table(Default::default());
    for path in paths {
        let value = read_toml_file(path)?;
        merge_value(&mut merged, value);
    }
    if let Value::Table(table) = merged {
        if table.is_empty() {
            return Ok(AppSettings::default());
        }
        let settings: AppSettings = Value::Table(table).try_into()?;
        return Ok(settings);
    }
    Ok(AppSettings::default())
}

fn read_toml_file(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(Value::Table(Default::default()));
    }
    let text = fs::read_to_string(path)?;
    Ok(text.parse::<Value>()?)
}

fn merge_value(base: &mut Value, override_value: Value) {
    match (base, override_value) {
        (Value::Table(base_table), Value::Table(override_table)) => {
            for (key, value) in override_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => merge_value(base_value, value),
                    None => {
                        base_table.insert(key, value);
                    }
                }
            }
        }
        (base_slot, value) => {
            *base_slot = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn returns_defaults_when_config_files_are_missing() {
        let settings = build_settings_from_paths(&[PathBuf::from("missing.toml")]).unwrap();

        assert_eq!(settings.llm.model, "ollama/gemma4:latest");
        assert_eq!(settings.builder.refine_split_threshold, 400);
        assert_eq!(settings.store.data_dir, PathBuf::from(".treefyust-store"));
    }

    #[test]
    fn merges_settings_files_in_order() {
        let tempdir = tempdir().unwrap();
        let base = tempdir.path().join("settings.toml");
        let local = tempdir.path().join("settings.local.toml");
        fs::write(
            &base,
            r#"
[llm]
model = "ollama/base"
base_url = "http://127.0.0.1:11434"
temperature = 0.3
max_tokens = 128

[builder]
refine_split_threshold = 100
refine_max_parts = 2

[store]
data_dir = "base-store"
"#,
        )
        .unwrap();
        fs::write(
            &local,
            r#"
[llm]
model = "ollama/local"

[store]
data_dir = "local-store"
"#,
        )
        .unwrap();

        let settings = build_settings_from_paths(&[base, local]).unwrap();

        assert_eq!(settings.llm.model, "ollama/local");
        assert_eq!(settings.llm.temperature, 0.3);
        assert_eq!(settings.builder.refine_max_parts, 2);
        assert_eq!(settings.store.data_dir, PathBuf::from("local-store"));
    }

    #[test]
    fn partial_section_uses_defaults_for_missing_fields() {
        let tempdir = tempdir().unwrap();
        let path = tempdir.path().join("settings.toml");
        fs::write(
            &path,
            r#"
[llm]
model = "ollama/custom"
"#,
        )
        .unwrap();

        let settings = build_settings_from_paths(&[path]).unwrap();

        assert_eq!(settings.llm.model, "ollama/custom");
        assert_eq!(
            settings.llm.base_url.as_deref(),
            Some("http://127.0.0.1:11434")
        );
        assert_eq!(settings.builder.refine_split_threshold, 400);
        assert_eq!(settings.store.data_dir, PathBuf::from(".treefyust-store"));
    }
}
