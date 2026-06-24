use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

use crate::model::Tree;
use crate::query::TreeIndex;

#[derive(Debug, Clone)]
pub struct RegistryStore {
    pub data_dir: PathBuf,
}

impl RegistryStore {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
        }
    }

    pub fn trees_dir(&self) -> PathBuf {
        self.data_dir.join("trees")
    }

    pub fn indexes_dir(&self) -> PathBuf {
        self.data_dir.join("indexes")
    }

    pub fn builds_dir(&self) -> PathBuf {
        self.data_dir.join("builds")
    }

    pub fn originals_dir(&self) -> PathBuf {
        self.data_dir.join("originals")
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.data_dir.join("sessions")
    }

    pub fn queries_path(&self) -> PathBuf {
        self.data_dir.join("queries.jsonl")
    }

    pub fn tree_path(&self, tree_id: &str) -> PathBuf {
        self.trees_dir().join(format!("{tree_id}.json"))
    }

    pub fn index_path(&self, tree_id: &str) -> PathBuf {
        self.indexes_dir().join(format!("{tree_id}.json"))
    }

    pub fn build_path(&self, build_id: &str) -> PathBuf {
        self.builds_dir().join(format!("{build_id}.json"))
    }

    pub fn session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir().join(format!("{session_id}.json"))
    }

    pub fn original_path(&self, storage_key: &str) -> PathBuf {
        self.originals_dir().join(storage_key)
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        fs::create_dir_all(self.trees_dir())?;
        fs::create_dir_all(self.indexes_dir())?;
        fs::create_dir_all(self.builds_dir())?;
        fs::create_dir_all(self.originals_dir())?;
        fs::create_dir_all(self.sessions_dir())?;
        Ok(())
    }

    pub fn save_tree(&self, tree: &Tree) -> Result<PathBuf> {
        self.ensure_dirs()?;
        let path = self.tree_path(&tree.node_id);
        write_json_atomically(&path, tree)?;
        Ok(path)
    }

    pub fn save_index(&self, index: &TreeIndex) -> Result<PathBuf> {
        self.ensure_dirs()?;
        let path = self.index_path(&index.tree_id);
        write_json_atomically(&path, index)?;
        Ok(path)
    }

    pub fn save_build<T: Serialize>(&self, build_id: &str, build: &T) -> Result<PathBuf> {
        self.ensure_dirs()?;
        let path = self.build_path(build_id);
        write_json_atomically(&path, build)?;
        Ok(path)
    }

    pub fn load_builds(&self) -> Result<HashMap<String, Value>> {
        let mut builds = HashMap::new();
        if !self.builds_dir().exists() {
            return Ok(builds);
        }
        for path in json_files(&self.builds_dir())? {
            let text = fs::read_to_string(path)?;
            let build: Value = serde_json::from_str(&text)?;
            if let Some(id) = build.get("id").and_then(Value::as_str) {
                builds.insert(id.to_string(), build);
            }
        }
        Ok(builds)
    }

    pub fn delete_build(&self, build_id: &str) -> Result<()> {
        remove_if_exists(self.build_path(build_id))
    }

    pub fn save_original(&self, build_id: &str, filename: &str, data: &[u8]) -> Result<String> {
        self.ensure_dirs()?;
        let safe_name = Path::new(filename)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("document.bin");
        let relative_key = format!("{build_id}/{safe_name}");
        let path = self.original_path(&relative_key);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        write_bytes_atomically(&path, data)?;
        Ok(relative_key)
    }

    pub fn delete_originals(&self, build_id: &str) -> Result<()> {
        let path = self.originals_dir().join(build_id);
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
        Ok(())
    }

    pub fn append_query<T: Serialize>(&self, query: &T) -> Result<()> {
        self.ensure_dirs()?;
        let mut file = File::options()
            .create(true)
            .append(true)
            .open(self.queries_path())?;
        file.write_all(serde_json::to_string(query)?.as_bytes())?;
        file.write_all(b"\n")?;
        Ok(())
    }

    pub fn load_queries(&self, limit: usize) -> Result<Vec<Value>> {
        let path = self.queries_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(path)?;
        let mut queries = Vec::new();
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            queries.push(serde_json::from_str(line)?);
        }
        let start = queries.len().saturating_sub(limit);
        let mut recent = queries.split_off(start);
        recent.reverse();
        Ok(recent)
    }

    pub fn save_session<T: Serialize>(&self, session_id: &str, session: &T) -> Result<PathBuf> {
        self.ensure_dirs()?;
        let path = self.session_path(session_id);
        write_json_atomically(&path, session)?;
        Ok(path)
    }

    pub fn load_sessions(&self) -> Result<HashMap<String, Value>> {
        let mut sessions = HashMap::new();
        if !self.sessions_dir().exists() {
            return Ok(sessions);
        }
        for path in json_files(&self.sessions_dir())? {
            let text = fs::read_to_string(path)?;
            let session: Value = serde_json::from_str(&text)?;
            if let Some(id) = session.get("session_id").and_then(Value::as_str) {
                sessions.insert(id.to_string(), session);
            }
        }
        Ok(sessions)
    }

    pub fn delete_session(&self, session_id: &str) -> Result<bool> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(false);
        }
        fs::remove_file(path)?;
        Ok(true)
    }

    pub fn load_trees(&self) -> Result<HashMap<String, Tree>> {
        let mut trees = HashMap::new();
        if !self.trees_dir().exists() {
            return Ok(trees);
        }
        for path in json_files(&self.trees_dir())? {
            let text = fs::read_to_string(path)?;
            let tree: Tree = serde_json::from_str(&text)?;
            trees.insert(tree.node_id.clone(), tree);
        }
        Ok(trees)
    }

    pub fn load_indexes(
        &self,
        tree_ids: Option<&std::collections::HashSet<String>>,
    ) -> Result<HashMap<String, TreeIndex>> {
        let mut indexes = HashMap::new();
        if !self.indexes_dir().exists() {
            return Ok(indexes);
        }
        for path in json_files(&self.indexes_dir())? {
            let text = fs::read_to_string(path)?;
            let index: TreeIndex = serde_json::from_str(&text)?;
            if tree_ids.is_some_and(|ids| !ids.contains(&index.tree_id)) {
                continue;
            }
            indexes.insert(index.tree_id.clone(), index);
        }
        Ok(indexes)
    }

    pub fn delete_tree(&self, tree_id: &str) -> Result<()> {
        remove_if_exists(self.tree_path(tree_id))
    }

    pub fn delete_index(&self, tree_id: &str) -> Result<()> {
        remove_if_exists(self.index_path(tree_id))
    }

    pub fn delete_bundle(&self, tree_id: &str) -> Result<()> {
        self.delete_tree(tree_id)?;
        self.delete_index(tree_id)?;
        self.delete_build(tree_id)?;
        self.delete_originals(tree_id)
    }
}

fn json_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) == Some("json") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn remove_if_exists(path: PathBuf) -> Result<()> {
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn write_json_atomically<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    write_bytes_atomically(path, serde_json::to_string_pretty(value)?.as_bytes())
}

fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp_path = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("registry"),
        Uuid::new_v4().simple()
    ));
    let mut file = File::create(&temp_path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temp_path, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use crate::builder::{BuildOptions, build_tree_from_text};
    use crate::query::build_tree_index;

    use super::*;

    #[test]
    fn saves_loads_and_deletes_tree_index() {
        let tempdir = tempdir().unwrap();
        let store = RegistryStore::new(tempdir.path());
        let mut tree = build_tree_from_text(
            "# Intro\n\nHello world.",
            "saved.md",
            BuildOptions::default(),
        );
        tree.node_id = "tree-1".to_string();
        let index = build_tree_index(&tree);

        assert!(store.save_tree(&tree).unwrap().exists());
        assert!(store.save_index(&index).unwrap().exists());
        assert_eq!(store.load_trees().unwrap()["tree-1"].title, "saved.md");
        assert_eq!(
            store.load_indexes(None).unwrap()["tree-1"].tree_title,
            "saved.md"
        );

        store.delete_bundle("tree-1").unwrap();
        assert!(!store.tree_path("tree-1").exists());
        assert!(!store.index_path("tree-1").exists());
    }
}
