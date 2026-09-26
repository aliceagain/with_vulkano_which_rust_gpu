use std::sync::Arc;

use glam::UVec2;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{
    CommandBufferUsage, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineShaderStageCreateInfo,
};

use crate::errors::RenderErr;
use crate::render::{Render, image_from_buffer};

fn build_command_buffer(
    render: &Render,
    work_group: &[u32; 3],
    pipeline: &Arc<ComputePipeline>,
    set: &Arc<DescriptorSet>,
    image: &Arc<Image>,
    buf: &Subbuffer<[u8]>,
) -> Result<Arc<PrimaryAutoCommandBuffer>, RenderErr> {
    let mut builder = render.new_cmd_buffer_builder(CommandBufferUsage::OneTimeSubmit)?;

    builder
        .bind_pipeline_compute(pipeline.clone())?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set.clone(),
        )?;

    unsafe {
        builder.dispatch(*work_group)?;
    }
    builder
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))?;
    Ok(builder.build()?)
}

fn compute_pipeline(render: &Render) -> Result<Arc<ComputePipeline>, RenderErr> {
    const MANDELBROT: &[u8] = include_bytes!(env!("mandelbrot.spv"));

    let shader = render.new_shader(MANDELBROT)?;

    let cs = shader
        .entry_point("main_cs")
        .map_or_else(|| Err(RenderErr::new("failed to get entry_point")), Ok)?;
    let stage_info = PipelineShaderStageCreateInfo::new(cs);
    let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_info])
        .into_pipeline_layout_create_info(render.device.clone())?;

    render.new_compute_pipeline(stage_info, layout_info)
}

fn descriptor_set(
    render: &Render,
    pipeline: &Arc<ComputePipeline>,
    image: &Arc<Image>,
) -> Result<Arc<DescriptorSet>, RenderErr> {
    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = pipeline
        .layout()
        .set_layouts()
        .get(descriptor_set_layout_index)
        .map_or_else(|| Err(RenderErr::new("No Descriptor Set Layouts")), Ok)?;

    let view = ImageView::new_default(image.clone())?;

    Ok(DescriptorSet::new(
        render.desc_set_alloc.clone(),
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::image_view(0, view)],
        [],
    )?)
}

pub fn mandelbrot(render: &Render) -> Result<(), RenderErr> {
    const IMAGE_SIZE: u32 = 4096;
    const LOCAL_SIZE: u32 = 8;
    const IMAGE_SIZE_BYTES: u32 = IMAGE_SIZE * IMAGE_SIZE * 4;
    const WORK_GROUP: [u32; 3] = [IMAGE_SIZE / LOCAL_SIZE, IMAGE_SIZE / LOCAL_SIZE, 1];

    let image = render.new_image_buf(
        UVec2::from_array([IMAGE_SIZE, IMAGE_SIZE]),
        ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
    )?;

    let compute_pipeline = compute_pipeline(render)?;
    let descriptor_set = descriptor_set(render, &compute_pipeline, &image)?;

    let buf = render.new_host_dst_buf(IMAGE_SIZE_BYTES)?;

    let command_buffer = build_command_buffer(
        render,
        &WORK_GROUP,
        &compute_pipeline,
        &descriptor_set,
        &image,
        &buf,
    )?;

    render.execute_and_wait(command_buffer)?;

    let buffer_content = buf.read()?;
    image_from_buffer(&buffer_content, UVec2::splat(IMAGE_SIZE))?.save("mandelbrot.png")?;

    Ok(())
}
