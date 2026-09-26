#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{UVec2, UVec3, Vec2, Vec3};
use spirv_std::{Image, glam, spirv};

pub fn fractal(id: UVec3, image: &Image!(2D, format = rgba8, sampled = false, depth = false)) {
    let norm_coordinates =
        (id.truncate().as_vec2() / image.query_size::<UVec2>().as_vec2()) - Vec2 { x: 0.5, y: 0.0 };
    let c = Vec2 {
        x: norm_coordinates.x * 2.97 - 0.72,
        y: norm_coordinates.y * 2.84 - 1.37,
    };

    let mut z = Vec2::ZERO;
    let mut i = 0.0;
    let iter = image.query_size::<UVec2>().x >> 3;
    for _ in 0..iter {
        z = Vec2 {
            x: z.x * z.x - z.y * z.y + c.x,
            y: z.y * z.x + z.x * z.y + c.y,
        };

        if z.length() > 4.0 {
            break;
        }
        i += 1.0 / iter as f32;
    }

    let color = UVec3 {
        x: 0x78,
        y: 0x51,
        z: 0xa9,
    }
    .as_vec3()
        / 255.0;
    let out = if i > 0.99 {
        Vec3::ZERO
    } else {
        let lerp = i * (z.length() / 4.0);
        color.lerp(Vec3::ONE, lerp * 16.0)
    }
    .extend(1.0);

    unsafe {
        image.write(id.truncate(), out);
    }
}

#[spirv(compute(threads(8, 8, 1)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] image_out: &Image!(
        2D,
        format = rgba8,
        sampled = false,
        depth = false
    ),
) {
    fractal(id, image_out);
}
