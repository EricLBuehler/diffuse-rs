use diffusers_core::{DiffusionGenerationParams, ModelPaths, Pipeline, TokenSource};

fn main() -> anyhow::Result<()> {
    let pipeline = Pipeline::load(
        ModelPaths::from_model_id("black-forest-labs/FLUX.1-schnell"),
        false,
        TokenSource::CacheToken,
        None,
    )?;

    let images = pipeline.forward(
        vec!["Draw a picture of a beautiful sunset in the winter in the mountains.".to_string()],
        DiffusionGenerationParams::default(),
    )?;
    images[0].save("image.png")?;

    Ok(())
}
