use glam::UVec2;
use vulkano::command_buffer::{CommandBufferUsage, CopyImageToBufferInfo, SubpassEndInfo};
use vulkano::format::Format;
use vulkano::image::ImageUsage;
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;

use crate::errors::RenderErr;
use crate::render::{Render, Vert, image_from_buffer, new_framebuffer};

pub fn graphics(render: &Render) -> Result<(), RenderErr> {
    const VERTICES: &[u8] = include_bytes!(env!("vertices.spv"));

    // Shader Entrypoints
    let shader = render.new_shader(VERTICES)?;
    let vs = shader.entry_point("vs_main").map_or_else(
        || {
            Err(RenderErr::new(
                "Didn't find vs_main entry point for shader.",
            ))
        },
        Ok,
    )?;
    let fs = shader.entry_point("fs_main").map_or_else(
        || {
            Err(RenderErr::new(
                "Didn't find fs_main entry point for shader.",
            ))
        },
        Ok,
    )?;

    // Image Buffer
    let image_buf = render.new_image_buf(
        UVec2 { x: 1024, y: 1024 },
        ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
    )?;

    // Vertex Buffer
    let vertex1 = Vert { vert: [-0.5, -0.5] };
    let vertex2 = Vert { vert: [0.0, 0.5] };
    let vertex3 = Vert { vert: [0.5, -0.25] };
    let vert_buf = render.new_vertex_buf(vec![vertex1, vertex2, vertex3])?;

    // Vertex Input State
    let vertex_input_state = Vert::per_vertex()
        .definition(&vs)?;

    // Host Dst Buffer
    let buf = render.new_host_dst_buf(1024 * 1024 * 4)?;

    let render_pass = render.new_render_pass(Format::R8G8B8A8_UNORM)?;

    // Framebuffer
    let framebuffer = new_framebuffer(&render_pass, std::slice::from_ref(&image_buf))?
        .first()
        .map_or_else(|| Err(RenderErr::new("No framebuffers returned")), Ok)?
        .to_owned();

    // Graphics Pipeline Stages
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    // Graphics Pipeline Layout
    let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
        .into_pipeline_layout_create_info(render.device.clone())?;

    // Graphics Pipeline
    let pipeline = render.new_graphics_pipeline(
        Viewport {
            offset: [0.0, 0.0],
            extent: [1024.0, 1024.0],
            depth_range: 0.0..=1.0,
        },
        &render_pass,
        &stages,
        layout_info,
        vertex_input_state,
    )?;

    let command_buffer = {
        // Graphics Start :)
        let mut builder =
            render.start_render_pass(pipeline, framebuffer, CommandBufferUsage::OneTimeSubmit)?;
        builder
            .bind_vertex_buffers(0, vert_buf)?;

        // Draw Graphics
        unsafe { builder.draw(3, 1, 0, 0) }?;

        // Graphics done :)
        builder
            .end_render_pass(SubpassEndInfo::default())?
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image_buf, buf.clone()))?;

        // Build Command Buffer
        builder.build()
    }?;

    // Render
    render.execute_and_wait(command_buffer)?;

    // Copy Image to Hosts
    let buffer_content = buf.read()?;
    image_from_buffer(&buffer_content, UVec2::splat(1024))?.save("graphics.png")?;

    Ok(())
}
