use std::{fs, io};

use spirv_builder::Capability::ImageQuery;
use spirv_builder::{SpirvBuilder, SpirvMetadata};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shaders = fs::read_dir("shaders")?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, io::Error>>()?;

    for shader in shaders {
        let mut builder = SpirvBuilder::new(shader.as_path(), "spirv-unknown-vulkan1.4")
            .spirv_metadata(SpirvMetadata::Full)
            .capability(ImageQuery);
        builder.build_script.defaults = true;
        builder.build_script.env_shader_spv_path = Some(true);
        builder.build()?;
    }

    Ok(())
}
