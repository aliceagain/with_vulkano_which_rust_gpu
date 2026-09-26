use std::sync::Arc;

use glam::UVec2;
use image::{ImageBuffer, Rgba};
use vulkano::VulkanLibrary;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo, spirv};
use vulkano::swapchain::Surface;
use vulkano::sync::{self, GpuFuture};
use winit::event_loop::EventLoop;

use crate::errors::RenderErr;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Vert {
    #[format(R32G32_SFLOAT)]
    pub vert: [f32; 2],
}

#[derive(Clone)]
pub struct Render {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub queue_family_index: u32,
    pub mem_alloc: Arc<StandardMemoryAllocator>,
    pub desc_set_alloc: Arc<StandardDescriptorSetAllocator>,
    pub cmd_buf_alloc: Arc<StandardCommandBufferAllocator>,
}

impl Render {
    pub fn new(event_loop: Option<&EventLoop<()>>) -> Result<Self, RenderErr> {
        let library = VulkanLibrary::new()?;

        let required_extensions = match event_loop {
            Some(event_loop) => Surface::required_extensions(event_loop)?,
            None => InstanceExtensions::default(),
        };

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to create library Instance"))?;

        let extensions = DeviceExtensions {
            khr_swapchain: event_loop.is_some(),
            khr_vulkan_memory_model: true,
            ..DeviceExtensions::empty()
        };

        let features = DeviceFeatures {
            vulkan_memory_model: true,
            ..DeviceFeatures::empty()
        };

        let physical_device = pick_preferred_physical_device(&instance, &extensions, event_loop)?;
        let queue_family_index: u32 = u32::try_from(pick_queue_family(&physical_device)?)
            .map_err(|e| RenderErr::from(e).add_ctx("queue_family_index exceeds u32"))?;

        let (device, queue) =
            new_device(physical_device, queue_family_index, &extensions, &features)?;

        let mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let desc_set_alloc = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));
        let cmd_buf_alloc = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        Ok(Self {
            instance,
            device,
            queue,
            queue_family_index,
            mem_alloc,
            desc_set_alloc,
            cmd_buf_alloc,
        })
    }

    pub fn new_cmd_buffer_builder(
        &self,
        usage: CommandBufferUsage,
    ) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, RenderErr> {
        AutoCommandBufferBuilder::primary(
            self.cmd_buf_alloc.clone(),
            self.queue_family_index,
            usage,
        )
        .map_err(|e| RenderErr::from(e).add_ctx("Could not create command buffer builder"))
    }

    pub fn new_shader(&self, shader_bytes: &[u8]) -> Result<Arc<ShaderModule>, RenderErr> {
        let shader_words = spirv::bytes_to_words(shader_bytes)?;
        let info = ShaderModuleCreateInfo::new(&shader_words);

        unsafe { ShaderModule::new(self.device.clone(), info) }
            .map_err(|e| RenderErr::from(e).add_ctx("failed to create shader module"))
    }

    pub fn new_compute_pipeline(
        &self,
        stage_info: PipelineShaderStageCreateInfo,
        layout_info: PipelineLayoutCreateInfo,
    ) -> Result<Arc<ComputePipeline>, RenderErr> {
        let layout = PipelineLayout::new(self.device.clone(), layout_info)
            .map_err(|e| RenderErr::from(e).add_ctx("Failed to create pipeline layout"))?;

        ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage_info, layout),
        )
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to create compute pipeline"))
    }

    pub fn new_graphics_pipeline(
        &self,
        viewport: Viewport,
        render_pass: &Arc<RenderPass>,
        stages: &[PipelineShaderStageCreateInfo],
        layout_info: PipelineLayoutCreateInfo,
        vertex_input_state: VertexInputState,
    ) -> Result<Arc<GraphicsPipeline>, RenderErr> {
        let layout = PipelineLayout::new(self.device.clone(), layout_info)
            .map_err(|e| RenderErr::from(e).add_ctx("Failed to create Graphics pipeline layout"))?;

        let subpass = Subpass::from(render_pass.clone(), 0).map_or_else(
            || Err(RenderErr::new("Failed to create render subpass")),
            Ok,
        )?;

        GraphicsPipeline::new(
            self.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                // The stages of our pipeline, we have vertex and fragment stages.
                stages: stages.into(),
                // Describes the layout of the vertex input and how should it behave.
                vertex_input_state: Some(vertex_input_state),
                // Indicate the type of the primitives (the default is a list of triangles).
                input_assembly_state: Some(InputAssemblyState::default()),
                // Set the fixed viewport.
                viewport_state: Some(ViewportState {
                    viewports: std::iter::once(viewport).collect(),
                    ..Default::default()
                }),
                // Ignore these for now.
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                // This graphics pipeline object concerns the first pass of the render pass.
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .map_err(|e| RenderErr::from(e).add_ctx("Failed create Graphics Pipeline"))
    }

    pub fn execute_and_wait(
        &self,
        cmd_buf: Arc<PrimaryAutoCommandBuffer>,
    ) -> Result<(), RenderErr> {
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cmd_buf)?
            .then_signal_fence_and_flush()
            .map_err(|e| RenderErr::from(e).add_ctx("Failed to signal fence and flush"))?;

        future
            .wait(None)
            .map_err(|e| RenderErr::from(e).add_ctx("Failed waiting for future"))
    }

    pub fn new_image_buf(&self, size: UVec2, usage: ImageUsage) -> Result<Arc<Image>, RenderErr> {
        Ok(Image::new(
            self.mem_alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: size.extend(1).into(),
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?)
    }

    pub fn new_host_dst_buf(&self, size: u32) -> Result<Subbuffer<[u8]>, RenderErr> {
        Ok(Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..size).map(|_| 0u8),
        )?)
    }

    pub fn new_vertex_buf(&self, vertices: Vec<Vert>) -> Result<Subbuffer<[Vert]>, RenderErr> {
        Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to create vertex buffer"))
    }

    pub fn new_render_pass(&self, format: Format) -> Result<Arc<RenderPass>, RenderErr> {
        vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                format: format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to create render pass"))
    }

    pub fn start_render_pass(
        &self,
        pipeline: Arc<GraphicsPipeline>,
        framebuffer: Arc<Framebuffer>,
        usage: CommandBufferUsage,
    ) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, RenderErr> {
        let mut builder = self.new_cmd_buffer_builder(usage)?;
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .map_err(|e| RenderErr::from(e).add_ctx("Failed to begin Render Pass"))?
            // Graphics :)
            .bind_pipeline_graphics(pipeline)
            .map_err(|e| RenderErr::from(e).add_ctx("Failed to bind graphics pipeline"))?;
        Ok(builder)
    }
}

pub fn image_from_buffer(
    buf: &[u8],
    image_size: UVec2,
) -> Result<image::ImageBuffer<image::Rgba<u8>, &[u8]>, RenderErr> {
    ImageBuffer::<Rgba<u8>, _>::from_raw(image_size.x, image_size.y, buf).map_or_else(
        || {
            Err(RenderErr::new(
                "ImageBuffer::from_raw failed: container not big enough.",
            ))
        },
        Ok,
    )
}

pub fn new_framebuffer(
    render_pass: &Arc<RenderPass>,
    images: &[Arc<Image>],
) -> Result<Vec<Arc<Framebuffer>>, RenderErr> {
    images
        .iter()
        .map(|i| {
            let view = ImageView::new_default(i.clone())
                .map_err(|e| RenderErr::from(e).add_ctx("Failed to create image view"))?;
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .map_err(|e| RenderErr::from(e).add_ctx("Failed to create framebuffer"))
        })
        .try_collect()
}

fn pick_preferred_physical_device(
    instance: &Arc<Instance>,
    device_extensions: &DeviceExtensions,
    event_loop: Option<&EventLoop<()>>,
) -> Result<Arc<PhysicalDevice>, RenderErr> {
    instance
        .enumerate_physical_devices()
        .map_err(|e| RenderErr::from(e).add_ctx("Could not enumerate devices"))?
        .filter(|d| d.supported_extensions().contains(device_extensions))
        .filter(|d| {
            d.queue_family_properties()
                .iter()
                .zip(0u32..)
                .any(|(q, i)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && event_loop.is_none_or(|event_loop| {
                            d.presentation_support(i, event_loop).is_ok_and(|s| s)
                        })
                })
        })
        .min_by_key(|d| match d.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .map_or_else(|| Err(RenderErr::new("No devices available")), Ok)
}

fn pick_queue_family(physical_device: &Arc<PhysicalDevice>) -> Result<usize, RenderErr> {
    physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .map_or_else(
            || Err(RenderErr::new("Couldn't find a graphical queue family")),
            Ok,
        )
}

fn new_device(
    physical_device: Arc<PhysicalDevice>,
    queue_family_index: u32,
    extensions: &DeviceExtensions,
    features: &DeviceFeatures,
) -> Result<(Arc<Device>, Arc<Queue>), RenderErr> {
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: *extensions,
            enabled_features: *features,
            ..Default::default()
        },
    )
    .map_err(|e| RenderErr::from(e).add_ctx("Failed to create device"))?;

    // Just use the first queue for now.
    let queue = queues
        .next()
        .map_or_else(|| Err(RenderErr::new("No queues available?")), Ok)?;

    Ok((device, queue))
}
