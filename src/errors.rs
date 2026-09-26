use std::any::type_name;
use std::backtrace::Backtrace;
use std::error::Error;
use std::fmt;
use std::num::TryFromIntError;
use std::panic::Location;

use image::ImageError;
use vulkano::command_buffer::CommandBufferExecError;
use vulkano::pipeline::layout::IntoPipelineLayoutCreateInfoError;
use vulkano::shader::spirv::SpirvBytesNotMultipleOf4;
use vulkano::swapchain::FromWindowError;
use vulkano::sync::HostAccessError;
use vulkano::{LoadingError, Validated, VulkanError};
use winit::error::{EventLoopError, OsError};
use winit::raw_window_handle::HandleError;

use crate::errors::RenderErr::{OtherError, RebuildSwapchain};

#[derive(Debug)]
pub enum RenderErr {
    RebuildSwapchain,
    OtherError {
        err: String,
        ctx: String,
        loc: String,
        bt: Backtrace,
    },
}

impl RenderErr {
    #[track_caller]
    #[inline]
    pub fn new(err: &str) -> Self {
        let loc = Location::caller();
        OtherError {
            err: err.to_string(),
            ctx: String::new(),
            loc: format!("{0}:{1}:{2}", loc.file(), loc.line(), loc.column()),
            bt: Backtrace::capture(),
        }
    }

    pub fn add_ctx(mut self, add_context: &str) -> Self {
        match &mut self {
            OtherError { ctx, .. } => ctx.push_str(add_context),
            RebuildSwapchain => {}
        }
        self
    }
}

trait GenericRenderErrFrom {}
impl GenericRenderErrFrom for LoadingError {}
impl GenericRenderErrFrom for HandleError {}
impl GenericRenderErrFrom for TryFromIntError {}
impl GenericRenderErrFrom for SpirvBytesNotMultipleOf4 {}
impl GenericRenderErrFrom for CommandBufferExecError {}
impl GenericRenderErrFrom for VulkanError{}
impl GenericRenderErrFrom for OsError{}
impl GenericRenderErrFrom for FromWindowError{}
impl GenericRenderErrFrom for EventLoopError{}
impl GenericRenderErrFrom for IntoPipelineLayoutCreateInfoError{}
impl GenericRenderErrFrom for HostAccessError{}
impl GenericRenderErrFrom for ImageError{}
impl<E> GenericRenderErrFrom for Box<E>{}

impl<E: Error + GenericRenderErrFrom> From<E> for RenderErr {
    #[track_caller]
    #[inline]
    fn from(e: E) -> Self {
        let loc = Location::caller();
        OtherError {
            err: e.to_string(),
            ctx: type_name::<E>().to_string(),
            loc: format!("{0}:{1}:{2}", loc.file(), loc.line(), loc.column()),
            bt: Backtrace::capture(),
        }
    }
}

impl<E: Error> From<Validated<E>> for RenderErr {
    #[track_caller]
    #[inline]
    fn from(e: Validated<E>) -> Self {
        let loc = Location::caller();
        OtherError {
            err: match e {
                Validated::Error(e) => e.to_string(),
                Validated::ValidationError(e) => e.to_string(),
            },
            ctx: type_name::<E>().to_string(),
            loc: format!("{0}:{1}:{2}", loc.file(), loc.line(), loc.column()),
            bt: Backtrace::capture(),
        }
    }
}

impl fmt::Display for RenderErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            RebuildSwapchain => write!(f, "Needs Redraw."),
            OtherError { err, ctx, loc, bt } => {
                writeln!(f, "Error({ctx}) [{loc}]: {err}\nbacktrace:\n{bt}")
            }
        }
    }
}
