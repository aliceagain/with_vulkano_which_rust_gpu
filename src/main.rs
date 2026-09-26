#![feature(iterator_try_collect)]

use crate::errors::RenderErr;
use crate::graphics::graphics;
use crate::mandelbrot::mandelbrot;
use crate::render::Render;
use crate::window::window;

mod errors;
mod graphics;
mod mandelbrot;
mod render;
mod window;
mod with;

fn main() -> Result<(), ()> {
    if let Err(e) = inner_main() {
        println!("{e}");
        Err(())
    } else {
        println!("Everything succeeded!");
        Ok(())
    }
}

fn inner_main() -> Result<(), RenderErr> {
    let render = Render::new(None)?;

    mandelbrot(&render)?;

    graphics(&render)?;

    window()?;

    Ok(())
}
