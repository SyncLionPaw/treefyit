use anyhow::Result;
use treefyust::config::get_settings;
use treefyust::server::serve;
use treefyust::store::RegistryStore;

#[tokio::main]
async fn main() -> Result<()> {
    let settings = get_settings()?;
    let store = RegistryStore::new(settings.store.data_dir);
    serve(([0, 0, 0, 0], 8765), Some(store)).await
}
