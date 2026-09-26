use std::collections::VecDeque;
use std::sync::Arc;

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassEndInfo,
};
use vulkano::device::{Device, Queue};
use vulkano::image::{Image, ImageUsage};
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::swapchain::{
    self, Surface, SurfaceInfo, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
    SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::errors::RenderErr;
use crate::render::{Render, Vert, new_framebuffer};

pub type Fence = FenceSignalFuture<Box<dyn GpuFuture>>;
pub type DrawCmdBufFn = fn(
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    resources: &RenderResources,
)
    -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, RenderErr>;

#[derive(Clone)]
pub struct RenderResources {
    pub vert_buf: Subbuffer<[Vert]>,
    pub vertex_input_state: VertexInputState,
    pub stages: [PipelineShaderStageCreateInfo; 2],
    pub layout_info: PipelineLayoutCreateInfo,
}

#[derive(Clone)]
pub struct Frame {
    pub fence: Arc<Fence>,
    cmd_buf: Arc<PrimaryAutoCommandBuffer>,
}

#[derive(Clone)]
struct Which {
    pub surface: Arc<Surface>,
    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<Image>>,
    pub dimensions: PhysicalSize<u32>,
    add_draw_calls: DrawCmdBufFn,
    frames: VecDeque<Frame>,
}

#[derive(Clone)]
pub struct With {
    render: Render,
    resources: RenderResources,
    which: Option<Which>,
}

impl With {
    pub const fn new(render: Render, resources: RenderResources) -> Self {
        Self {
            render,
            resources,
            which: None,
        }
    }

    pub fn create_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        add_draw_calls: DrawCmdBufFn,
    ) -> Result<(), RenderErr> {
        self.which = Some(Which::new(
            &self.render,
            &self.resources,
            event_loop,
            add_draw_calls,
        )?);

        Ok(())
    }

    pub fn resize(&mut self, dimensions: PhysicalSize<u32>) -> Result<(), RenderErr> {
        let Some(which) = &mut self.which else {
            return Err(RenderErr::new("resize called without calling add_window"));
        };

        which.resize(dimensions)?;
        which.build_frames(&self.render, &self.resources)
    }

    pub fn redraw(&mut self) -> Result<(), RenderErr> {
        let Some(which) = &mut self.which else {
            return Err(RenderErr::new("rebuild called without calling add_window"));
        };

        which.redraw(&self.render, &self.resources)
    }
}

impl Which {
    fn new(
        render: &Render,
        resources: &RenderResources,
        event_loop: &ActiveEventLoop,
        add_draw_calls: DrawCmdBufFn,
    ) -> Result<Self, RenderErr> {
        let window = event_loop
            .create_window(Window::default_attributes())
            .map(Arc::new)
            .map_err(|e| RenderErr::from(e).add_ctx("Failed to create Window"))?;

        let dimensions = window.inner_size();

        let surface = Surface::from_window(render.instance.clone(), window)
            .map_err(|e| RenderErr::from(e).add_ctx("Failed to create Surface from window"))?;

        let (swapchain, images) = new_swapchain(render, &surface, dimensions.into())?;

        let mut which = Self {
            surface,
            swapchain,
            images,
            dimensions,
            frames: VecDeque::new(),
            add_draw_calls,
        };

        which.build_frames(render, resources)?;

        Ok(which)
    }

    fn resize(&mut self, dimensions: PhysicalSize<u32>) -> Result<(), RenderErr> {
        self.dimensions = dimensions;
        (self.swapchain, self.images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                // Here, `image_extend` will correspond to the window dimensions.
                image_extent: self.dimensions.into(),
                ..self.swapchain.create_info()
            })
            .map_err(|e| RenderErr::from(e).add_ctx("Failed to rebuild swapchain"))?;
        Ok(())
    }

    fn rebuild(&mut self, render: &Render) -> Result<(), RenderErr> {
        (self.swapchain, self.images) =
            new_swapchain(render, &self.surface, self.dimensions.into())?;

        Ok(())
    }

    fn redraw(
        &mut self,
        render: &Render,
        resources: &RenderResources,
    ) -> Result<(), RenderErr> {
        let (Some(curr_frame), Some(back_frame)) = (self.frames.pop_front(), self.frames.back())
        else {
            return Err(RenderErr::new("Not enough frames"));
        };

        // Render current frame;
        curr_frame.fence.wait(None).map_err(|e| {
            RenderErr::from(e).add_ctx("Failed waiting for image fence to complete")
        })?;

        match self.rebuild_frame(render, &curr_frame, Some(&back_frame.fence.clone())) {
            Ok(frame) => {
                self.frames.push_back(frame);
                Ok(())
            }
            Err(RenderErr::RebuildSwapchain) => {
                self.rebuild(render)?;
                self.build_frames(render, resources)?;
                Ok(())
            }
            Err(e) => Err(e.add_ctx("Rebuilding frame future failed")),
        }
    }

    fn acquire_futures(&self) -> Result<Vec<SwapchainAcquireFuture>, RenderErr> {
        let mut acquire_futures = Vec::with_capacity(self.images.len());
        for _ in 0..self.images.len() {
            let (_, _, acquire_future) =
                swapchain::acquire_next_image(self.swapchain.clone(), None).map_err(|e| {
                    RenderErr::from(e).add_ctx("Failed to acquire swapchain image during init")
                })?;
            acquire_futures.push(acquire_future);
        }

        Ok(acquire_futures)
    }

    fn build_frames(
        &mut self,
        render: &Render,
        resources: &RenderResources,
    ) -> Result<(), RenderErr> {
        let acquire_futures = self.acquire_futures()?;

        let mut prev_fence: Option<Arc<Fence>> = None;

        let cmd_bufs = self.build_cmd_buffers(render, resources)?;

        self.frames = cmd_bufs
            .iter()
            .zip(acquire_futures)
            .enumerate()
            .map(|(i, (cmd_buf, acquire_future))| {
                let image_i = u32::try_from(i).unwrap_or_default();
                let frame = self.build_frame(
                    render,
                    image_i,
                    cmd_buf,
                    prev_fence.as_ref(),
                    acquire_future,
                )?;
                prev_fence = Some(frame.fence.clone());
                Ok::<Frame, RenderErr>(frame)
            })
            .try_collect()?;

        Ok(())
    }

    fn build_frame(
        &self,
        render: &Render,
        i: u32,
        cmd_buf: &Arc<PrimaryAutoCommandBuffer>,
        prev_fence: Option<&Arc<Fence>>,
        acquire_future: SwapchainAcquireFuture,
    ) -> Result<Frame, RenderErr> {
        let fence = build_future(
            &render.device,
            &render.queue,
            SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), i),
            prev_fence.cloned(),
            cmd_buf,
            acquire_future,
        )?;
        Ok::<_, RenderErr>(Frame {
            fence,
            cmd_buf: cmd_buf.clone(),
        })
    }

    fn rebuild_frame(
        &self,
        render: &Render,
        curr_frame: &Frame,
        back_fence: Option<&Arc<Fence>>,
    ) -> Result<Frame, RenderErr> {
        let (image_i, suboptimal, acquire_future) =
            swapchain::acquire_next_image(self.swapchain.clone(), None)
                .map_err(|_| RenderErr::RebuildSwapchain)?;

        // Build new future.
        let curr_frame = self.build_frame(
            render,
            image_i,
            &curr_frame.cmd_buf,
            back_fence,
            acquire_future,
        )?;

        if suboptimal {
            return Err(RenderErr::RebuildSwapchain);
        }
        Ok(curr_frame)
    }

    fn build_cmd_buffers(
        &self,
        render: &Render,
        resources: &RenderResources,
    ) -> Result<Vec<Arc<PrimaryAutoCommandBuffer>>, RenderErr> {
        let render_pass = render.new_render_pass(self.swapchain.image_format())?;

        // Framebuffer
        let framebuffers = new_framebuffer(&render_pass, &self.images)?;

        // Graphics Pipeline
        let pipeline = render.new_graphics_pipeline(
            Viewport {
                offset: [0.0, 0.0],
                extent: self.dimensions.into(),
                depth_range: 0.0..=1.0,
            },
            &render_pass,
            &resources.stages,
            resources.layout_info.clone(),
            resources.vertex_input_state.clone(),
        )?;

        framebuffers
            .iter()
            .map(|framebuffer| {
                // Graphics Start :)
                let mut builder = render.start_render_pass(
                    pipeline.clone(),
                    framebuffer.clone(),
                    CommandBufferUsage::MultipleSubmit,
                )?;

                // Draw Graphics
                builder = (self.add_draw_calls)(builder, resources)?;

                // Graphics done :)
                builder
                    .end_render_pass(SubpassEndInfo::default())
                    .map_err(|e| RenderErr::from(e).add_ctx("Failed to end Render Pass"))?;

                // Build Command Buffer
                builder
                    .build()
                    .map_err(|e| RenderErr::from(e).add_ctx("Failed to build command buffer"))
            })
            .try_collect()
    }
}

fn new_swapchain(
    render: &Render,
    surface: &Arc<Surface>,
    dimensions: [u32; 2],
) -> Result<(Arc<vulkano::swapchain::Swapchain>, Vec<Arc<Image>>), RenderErr> {
    let device = &render.device;
    let phys_device = device.physical_device();

    let caps = phys_device
        .surface_capabilities(surface, SurfaceInfo::default())
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to get surface capabilities"))?;

    let composite_alpha = caps
        .supported_composite_alpha
        .into_iter()
        .next()
        .map_or_else(|| Err(RenderErr::new("No transparency support?")), Ok)?;
    let image_format = phys_device
        .surface_formats(surface, SurfaceInfo::default())
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to get surface formats"))?
        .first()
        .map_or_else(|| Err(RenderErr::new("No surface formats?")), Ok)?
        .0;

    vulkano::swapchain::Swapchain::new(
        render.device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count.saturating_add(1),
            image_format,
            image_extent: dimensions,
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            ..Default::default()
        },
    )
    .map_err(|e| RenderErr::from(e).add_ctx("Failed to create swapchain"))
}

#[allow(clippy::arc_with_non_send_sync)]
fn build_future(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    swapchain_info: SwapchainPresentInfo,
    prev_fence: Option<Arc<Fence>>,
    cmd_buf: &Arc<PrimaryAutoCommandBuffer>,
    acquire_future: SwapchainAcquireFuture,
) -> Result<Arc<Fence>, RenderErr> {
    Ok(Arc::new(
        if let Some(prev_fence) = prev_fence {
            prev_fence
                .join(acquire_future)
                .then_execute(queue.clone(), cmd_buf.clone())
                .map_err(|e| RenderErr::from(e).add_ctx("Failed to execute command buffer"))?
                .then_swapchain_present(queue.clone(), swapchain_info)
                .boxed()
        } else {
            sync::now(device.clone())
                .join(acquire_future)
                .then_execute(queue.clone(), cmd_buf.clone())
                .map_err(|e| RenderErr::from(e).add_ctx("Failed to execute command buffer"))?
                .then_swapchain_present(queue.clone(), swapchain_info)
                .boxed()
        }
        .then_signal_fence_and_flush()
        .map_err(|e| match e {
            Validated::Error(VulkanError::OutOfDate) => RenderErr::RebuildSwapchain,
            _ => RenderErr::new("Failed to signal fence and flush"),
        })?,
    ))
}
