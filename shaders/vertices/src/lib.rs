#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{Vec2, Vec4};
use spirv_std::{glam, spirv};

#[spirv(vertex)]
pub fn vs_main(#[spirv(position)] pos: &mut Vec4, vert: Vec2) {
    *pos = glam::vec4(vert.x, vert.y, 0.0, 1.0);
}

#[spirv(fragment)]
pub fn fs_main(color: &mut Vec4) {
    *color = glam::vec4(1.0, 0.0, 0.0, 1.0);
}
