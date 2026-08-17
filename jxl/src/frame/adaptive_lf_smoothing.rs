// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::JxlParallelRunner;
use crate::error::Result;
use crate::headers::frame_header::FrameHeader;
use crate::image::Image;
use crate::render::buffer_splitter::OutputChannelSplitter;
use num_traits::abs;

#[allow(clippy::excessive_precision)]
const W_SIDE: f32 = 0.20345139757231578;
#[allow(clippy::excessive_precision)]
const W_CORNER: f32 = 0.0334829185968739;
const W_CENTER: f32 = 1.0 - 4.0 * (W_SIDE + W_CORNER);

fn compute_pixel_channel(
    dc_factor: f32,
    gap: f32,
    x: usize,
    row_top: &[f32],
    row: &[f32],
    row_bottom: &[f32],
) -> (f32, f32, f32) {
    let tl = row_top[x - 1];
    let tc = row_top[x];
    let tr = row_top[x + 1];
    let ml = row[x - 1];
    let mc = row[x];
    let mr = row[x + 1];
    let bl = row_bottom[x - 1];
    let bc = row_bottom[x];
    let br = row_bottom[x + 1];
    let corner = tl + tr + bl + br;
    let side = ml + mr + tc + bc;
    let sm = corner * W_CORNER + side * W_SIDE + mc * W_CENTER;
    (mc, sm, gap.max(abs((mc - sm) / dc_factor)))
}

/// Upsample one dimension by a factor of two.
///
/// The interpolation is:
///
///   dst[2*x]     = 1/4 * left + 3/4 * center
///   dst[2*x + 1] = 3/4 * center + 1/4 * right
///
/// The nearest available sample is replicated at the image boundary.
fn upsample_h2(src: &Image<f32>, dst_width: usize) -> Result<Image<f32>> {
    let (src_width, height) = src.size();
    let mut dst = Image::<f32>::new((dst_width, height))?;

    for y in 0..height {
        let src_row = src.row(y);
        let dst_row = dst.row_mut(y);

        for sx in 0..src_width {
            let x = sx << 1;

            if x >= dst_width {
                break;
            }

            let left = if sx == 0 {
                src_row[sx]
            } else {
                src_row[sx - 1]
            };

            let center = src_row[sx];

            let right = if sx + 1 < src_width {
                src_row[sx + 1]
            } else {
                center
            };

            dst_row[x] = 0.25 * left + 0.75 * center;

            if x + 1 < dst_width {
                dst_row[x + 1] = 0.75 * center + 0.25 * right;
            }
        }
    }

    Ok(dst)
}

fn upsample_v2(src: &Image<f32>, dst_height: usize) -> Result<Image<f32>> {
    let (width, src_height) = src.size();
    let mut dst = Image::<f32>::new((width, dst_height))?;

    for sy in 0..src_height {
        let y = sy << 1;

        if y >= dst_height {
            break;
        }

        let row_center = src.row(sy);

        let row_top = if sy == 0 {
            row_center
        } else {
            src.row(sy - 1)
        };

        let row_bottom = if sy + 1 < src_height {
            src.row(sy + 1)
        } else {
            row_center
        };

        {
            let row_out = dst.row_mut(y);

            for x in 0..width {
                row_out[x] = 0.25 * row_top[x] + 0.75 * row_center[x];
            }
        }

        if y + 1 < dst_height {
            let row_out = dst.row_mut(y + 1);

            for x in 0..width {
                row_out[x] = 0.75 * row_center[x] + 0.25 * row_bottom[x];
            }
        }
    }

    Ok(dst)
}

fn chroma_upsample_lf(
    lf_image: &mut Image<f32>,
    target_size: (usize, usize),
    hshift: u8,
    vshift: u8,
) -> Result<()> {
    debug_assert!(hshift != 0 || vshift != 0);

    let source_size = lf_image.size();
    let mut current =
        std::mem::replace(lf_image, Image::<f32>::new(source_size)?);

    let (target_width, target_height) = target_size;

    // Upsample horizontally one factor-of-two stage at a time.
    //
    // This is deliberately performed over the complete image rather than
    // independently for each LF group. The neighbour required by the
    // interpolation may belong to an adjacent LF group.
    for _ in 0..hshift {
        if current.size().0 < target_width {
            current = upsample_h2(&current, target_width)?;
        }
    }

    // Upsample vertically one factor-of-two stage at a time.
    for _ in 0..vshift {
        if current.size().1 < target_height {
            current = upsample_v2(&current, target_height)?;
        }
    }

    debug_assert_eq!(current.size(), target_size);

    *lf_image = current;

    Ok(())
}

// TODO(veluca): consider SIMDfying this.
pub fn adaptive_lf_smoothing(
    lf_factors: [f32; 3],
    frame_header: &FrameHeader,
    lf_image: &mut [Image<f32>; 3],
    parallel_runner: &mut dyn JxlParallelRunner,
) -> Result<()> {
    let (xsize, ysize) = lf_image[0].size();

    if ysize <= 2 || xsize <= 2 {
        return Ok(());
    }

    let shifts: [(u8, u8); 3] = std::array::from_fn(|i| {
        (
            frame_header.hshift(i) as u8,
            frame_header.vshift(i) as u8,
        )
    });

    // Adaptive LF smoothing operates on the common full-resolution LF grid.
    // Chroma LF planes are decoded at their native subsampled resolution,
    // so upsample them before entering the smoothing pass.
    for ch in 0..3 {
        let (hshift, vshift) = shifts[ch];

        if hshift != 0 || vshift != 0 {
            chroma_upsample_lf(
                &mut lf_image[ch],
                (xsize, ysize),
                hshift,
                vshift,
            )?;
        }
    }

    let mut smoothed0 = Image::<f32>::new((xsize, ysize))?;
    let mut smoothed1 = Image::<f32>::new((xsize, ysize))?;
    let mut smoothed2 = Image::<f32>::new((xsize, ysize))?;

    let splitter0 = OutputChannelSplitter::from_image(&mut smoothed0);
    let splitter1 = OutputChannelSplitter::from_image(&mut smoothed1);
    let splitter2 = OutputChannelSplitter::from_image(&mut smoothed2);

    let num_lf_groups = frame_header.num_lf_groups();

    parallel_runner.run(num_lf_groups, &|g| {
        let r = frame_header.lf_group_rect(g);
        let mut out_ref_0 = splitter0.borrow_typed_rect::<f32>(r);
        let mut out_ref_1 = splitter1.borrow_typed_rect::<f32>(r);
        let mut out_ref_2 = splitter2.borrow_typed_rect::<f32>(r);

        for ly in 0..r.size.1 {
            let gy = r.origin.1 + ly;
            let row_0 = out_ref_0.typed_row_mut::<f32>(ly);
            let row_1 = out_ref_1.typed_row_mut::<f32>(ly);
            let row_2 = out_ref_2.typed_row_mut::<f32>(ly);

            for lx in 0..r.size.0 {
                let gx = r.origin.0 + lx;

                if gy == 0 || gy == ysize - 1 || gx == 0 || gx == xsize - 1 {
                    row_0[lx] = lf_image[0].row(gy)[gx];
                    row_1[lx] = lf_image[1].row(gy)[gx];
                    row_2[lx] = lf_image[2].row(gy)[gx];
                    continue;
                }

                let gap = 0.5;

                let (mc_x, sm_x, gap) = compute_pixel_channel(
                    lf_factors[0],
                    gap,
                    gx,
                    lf_image[0].row(gy - 1),
                    lf_image[0].row(gy),
                    lf_image[0].row(gy + 1),
                );

                let (mc_y, sm_y, gap) = compute_pixel_channel(
                    lf_factors[1],
                    gap,
                    gx,
                    lf_image[1].row(gy - 1),
                    lf_image[1].row(gy),
                    lf_image[1].row(gy + 1),
                );

                let (mc_b, sm_b, gap) = compute_pixel_channel(
                    lf_factors[2],
                    gap,
                    gx,
                    lf_image[2].row(gy - 1),
                    lf_image[2].row(gy),
                    lf_image[2].row(gy + 1),
                );

                let factor = (3.0 - 4.0 * gap).max(0.0);

                row_0[lx] = (sm_x - mc_x) * factor + mc_x;
                row_1[lx] = (sm_y - mc_y) * factor + mc_y;
                row_2[lx] = (sm_b - mc_b) * factor + mc_b;
            }
        }

        Ok(())
    })?;

    drop(splitter0);
    drop(splitter1);
    drop(splitter2);

    *lf_image = [smoothed0, smoothed1, smoothed2];

    Ok(())
}
