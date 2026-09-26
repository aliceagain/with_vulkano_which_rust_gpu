use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

use crate::errors::RenderErr;
use crate::render::{Render, Vert};
use crate::with::{RenderResources, With};

const VERTICES: &[u8] = include_bytes!(env!("vertices.spv"));

struct Window {
    with: With,
    error: Result<(), RenderErr>,
}

impl ApplicationHandler for Window {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        match self.with.create_window(event_loop, add_draw_calls) {
            Ok(()) => {}
            Err(e) => {
                self.error = Err(e.add_ctx("Window Init failed"));
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match self.handle_event(event_loop, &event) {
            Ok(()) => {}
            Err(e) => {
                self.error = Err(e.add_ctx("Window Init failed"));
                event_loop.exit();
            }
        }
    }
}

impl Window {
    const fn new(render: Render, resources: RenderResources) -> Self {
        Self {
            with: With::new(render, resources),
            error: Ok(()),
        }
    }

    fn handle_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: &WindowEvent,
    ) -> Result<(), RenderErr> {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                Ok(())
            }
            WindowEvent::RedrawRequested => self.with.redraw(),
            WindowEvent::Resized(new_dimensions) => self.with.resize(*new_dimensions),
            _ => Ok(()),
        }
    }
}

fn add_draw_calls(
    mut builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    resources: &RenderResources,
) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, RenderErr> {
    builder
        .bind_vertex_buffers(0, resources.vert_buf.clone())
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to bind vertex buffers"))?;

    // Draw Graphics
    unsafe { builder.draw(6, 1, 0, 0) }
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to build draw command"))?;

    Ok(builder)
}

pub fn window() -> Result<(), RenderErr> {
    let event_loop =
        EventLoop::new().map_err(|e| RenderErr::from(e).add_ctx("Failed to create EventLoop"))?;
    let render = Render::new(Some(&event_loop))?;

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

    // Vertex Buffer
    let vertex1 = Vert { vert: [-0.5, -0.5] };
    let vertex2 = Vert { vert: [0.0, 0.5] };
    let vertex3 = Vert { vert: [0.5, -0.25] };
    let vertex4 = Vert { vert: [-0.8, 0.5] };
    let vertex5 = Vert { vert: [0.8, -0.1] };
    let vertex6 = Vert { vert: [0.1, 0.65] };
    let vert_buf =
        render.new_vertex_buf(vec![vertex1, vertex2, vertex3, vertex4, vertex5, vertex6])?;

    // Vertex Input State
    let vertex_input_state = Vert::per_vertex()
        .definition(&vs)
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to create Vertex Buffer Description"))?;

    // Graphics Pipeline Stages
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    // Graphics Pipeline Layout
    let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
        .into_pipeline_layout_create_info(render.device.clone())
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to create Layout info"))?;

    let mut app = Window::new(
        render,
        RenderResources {
            vert_buf,
            vertex_input_state,
            stages,
            layout_info,
        },
    );

    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop
        .run_app(&mut app)
        .map_err(|e| RenderErr::from(e).add_ctx("Failed to run event loop app"))?;

    app.error
}
