use std::borrow::Cow;

use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::{
    conv::ParamsConv2D,
    cpu_backend::{copy_strided_src_, Im2Col, Map1, Map2, MatMul},
    shape::dims4,
    Layout, Result, WithDType,
};

pub(super) struct Conv2D<'a>(pub(super) &'a crate::conv::ParamsConv2D);

#[allow(dead_code)]
enum Conv2dImpl {
    TiledIm2Col,
    FullIm2Col,
    Direct,
}

const DEFAULT_CONV2D_IMPL: Conv2dImpl = Conv2dImpl::TiledIm2Col;

impl Map2 for Conv2D<'_> {
    const OP: &'static str = "conv2d";
    fn f<T: WithDType + num_traits::Num + Copy + 'static>(
        &self,
        inp: &[T],
        inp_l: &Layout,
        k: &[T],
        k_l: &Layout,
    ) -> Result<Vec<T>> {
        let p = self.0;

        // Specialization: pick the best algorithm based on parameters.
        // 1x1 convolutions with stride=1, padding=0, dilation=1
        if p.k_h == 1 && p.k_w == 1 && p.stride == 1 && p.padding == 0 && p.dilation == 1 {
            return conv2d_1x1(p, inp, inp_l, k, k_l);
        } else if p.k_h == 1 && p.k_w == 1 {
            // Other 1x1 convolutions for now are assumed faster with full im2col,
            // although with large enough input size, tiled will start beating it.
            return conv2d_im2col_gemm(p, inp, inp_l, k, k_l);
        }
        // TODO other cases

        // No fast path, fallback to default general impl.
        match DEFAULT_CONV2D_IMPL {
            Conv2dImpl::TiledIm2Col => conv2d_tiled(p, inp, inp_l, k, k_l),
            Conv2dImpl::Direct => conv2d_direct(p, inp, inp_l, k, k_l),
            Conv2dImpl::FullIm2Col => conv2d_im2col_gemm(p, inp, inp_l, k, k_l),
        }
    }
}

/// Fast kernel for 1x1 convolutions with stride=1, padding=0, dilation=1
/// These are just matrix multiplications: [c_out, c_in] @ [c_in, b*h*w] -> [c_out, b*h*w].
fn conv2d_1x1<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv2D,
    inp: &[T],
    inp_l: &Layout,
    k: &[T],
    k_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let inp_stride = inp_l.stride();
    let (inp_s0, inp_s1, inp_s2, inp_s3) =
        (inp_stride[0], inp_stride[1], inp_stride[2], inp_stride[3]);
    let k = &k[k_l.start_offset()..];
    let k_stride = k_l.stride();
    let (k_s0, k_s1) = (k_stride[0], k_stride[1]);
    let (out_h, out_w) = (p.out_h(), p.out_w());

    let spatial_size = out_h * out_w;
    let dst = vec![T::zero(); p.b_size * p.c_out * spatial_size];
    let k_reshaped: Cow<[T]> = if k_s0 == p.c_in && k_s1 == 1 {
        // Already contiguous, use slice directly
        Cow::Borrowed(&k[..p.c_out * p.c_in])
    } else {
        // Reshape kernel to [c_out, c_in]
        let mut k_reshaped = Vec::with_capacity(p.c_out * p.c_in);
        (0..p.c_out).for_each(|c_out_idx| {
            (0..p.c_in).for_each(|c_in_idx| {
                let k_idx = c_out_idx * k_s0 + c_in_idx * k_s1;
                k_reshaped.push(k[k_idx]);
            });
        });
        Cow::Owned(k_reshaped)
    };
    let k_layout = Layout::contiguous((p.c_out, p.c_in));

    // Process each batch
    (0..p.b_size).into_par_iter().try_for_each(|b_idx| {
        // Reshape input to [c_in, h*w] for this batch
        let mut inp_reshaped = Vec::with_capacity(p.c_in * spatial_size);
        for c_in_idx in 0..p.c_in {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    let inp_idx =
                        b_idx * inp_s0 + c_in_idx * inp_s1 + h_idx * inp_s2 + w_idx * inp_s3;
                    inp_reshaped.push(inp[inp_idx]);
                }
            }
        }
        let inp_layout = Layout::contiguous((p.c_in, spatial_size));

        // Perform matmul: [c_out, c_in] @ [c_in, spatial_size] -> [c_out, spatial_size]
        let matmul = MatMul((1, p.c_out, spatial_size, p.c_in));
        let result = matmul.f(&k_reshaped, &k_layout, &inp_reshaped, &inp_layout)?;

        // Copy result to output
        let out_offset = b_idx * p.c_out * spatial_size;
        for (i, r) in result.iter().enumerate() {
            unsafe {
                let ptr = dst.as_ptr().add(out_offset + i) as *mut T;
                *ptr = *r;
            }
        }
        Ok::<(), crate::Error>(())
    })?;

    Ok(dst)
}

/// General tiled convolution implementation using gemm.
///
/// Similar to full im2col, but instead of materializing the full matrix, we process input/output in tiles, in parallel.
fn conv2d_tiled<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv2D,
    inp: &[T],
    inp_l: &Layout,
    k: &[T],
    k_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3) = dims4(inp_l.stride())?;
    let k = &k[k_l.start_offset()..];
    let (k_s0, k_s1, k_s2, k_s3) = dims4(k_l.stride())?;
    let (out_h, out_w) = (p.out_h(), p.out_w());

    // Output shape: [b_size, c_out, out_h, out_w].
    let dst = vec![T::zero(); p.b_size * p.c_out * out_h * out_w];

    // Pack NCHW input as NHWC for the tiled inner loop. Do not infer this from
    // strides: NCHW can have the same numeric strides as NHWC for shapes where
    // channels equals width, e.g. [1, 32, 32, 32].
    let cont_s0 = p.i_h * p.i_w * p.c_in;
    let cont_s1 = p.i_w * p.c_in;
    let cont_s2 = p.c_in;
    let inp_cont: Vec<T> = {
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_h * p.i_w];
        for b_idx in 0..p.b_size {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    for c_idx in 0..p.c_in {
                        let src_idx =
                            b_idx * inp_s0 + c_idx * inp_s1 + h_idx * inp_s2 + w_idx * inp_s3;
                        let dst_idx = b_idx * cont_s0 + h_idx * cont_s1 + w_idx * cont_s2 + c_idx;
                        inp_cont[dst_idx] = inp[src_idx]
                    }
                }
            }
        }
        inp_cont
    };

    // shape of k: [c_out, c_in, k_h, k_w]
    // strides of k: [k_s0, k_s1, k_s2, k_s3]
    // For matmul, we need flattened k in shape [c_out, k_h * k_w * c_in]
    // with stride [k_h * k_w * c_in, 1]
    let k_size = p.c_in * p.k_h * p.k_w;
    let mut k_flat = Vec::with_capacity(p.c_out * k_size);
    for dst_c_idx in 0..p.c_out {
        for kh in 0..p.k_h {
            for kw in 0..p.k_w {
                for c_in_idx in 0..p.c_in {
                    let k_idx = dst_c_idx * k_s0 + c_in_idx * k_s1 + kh * k_s2 + kw * k_s3;
                    k_flat.push(k[k_idx]);
                }
            }
        }
    }
    // k_layout: [c_out, k_size] with stride [k_size, 1]
    let k_layout = Layout::contiguous((p.c_out, k_size));

    // TILE_SIZE is number of output pixels (out_h * out_w) per tile.
    // Higher tile size can be faster due to better usage of gemm,
    // but lower tile sizes enable bigger parallelism across tiles.
    // This parameter is impactful and may be dynamic or even runtime tunable in the future.
    const TILE_SIZE: usize = 512;

    let total_out_pixels = out_h * out_w;

    // Process batches and tiles in parallel using rayon.
    (0..p.b_size).into_par_iter().try_for_each(|b_idx| {
        let inp_offset = b_idx * cont_s0;
        let out_batch_offset = b_idx * (p.c_out * out_h * out_w);

        let num_tiles = total_out_pixels.div_ceil(TILE_SIZE);
        (0..num_tiles).into_par_iter().try_for_each(|tile_idx| {
            // Determine actual tile size (may be smaller at the end) {
            let tile_start = tile_idx * TILE_SIZE;
            let tile_end = (tile_start + TILE_SIZE).min(total_out_pixels);
            let tile_size = tile_end - tile_start;

            // Precompute output coordinates.
            // Used in both im2col extraction and writing output.
            let out_coords: Vec<_> = (tile_start..tile_end)
                .map(|idx| (idx / out_w, idx % out_w))
                .collect();

            // Build im2col tile: [k_size, tile_size]
            // This represents the input patches needed for this tile of outputs
            let mut col_tile = vec![T::zero(); k_size * tile_size];

            for (tile_idx, (out_y, out_x)) in out_coords.iter().enumerate() {
                // Extract the im2col patch for this output position
                for c_in in 0..p.c_in {
                    let mut patch_offset = c_in;
                    for kh in 0..p.k_h {
                        let in_y =
                            (out_y * p.stride + kh * p.dilation) as isize - p.padding as isize;
                        if in_y < 0 || in_y >= p.i_h as isize {
                            // Padding: already zero
                            patch_offset += p.c_in * p.k_w;
                            continue;
                        }
                        for kw in 0..p.k_w {
                            let in_x =
                                (out_x * p.stride + kw * p.dilation) as isize - p.padding as isize;

                            if in_x >= 0 && in_x < p.i_w as isize {
                                let in_y = in_y as usize;
                                let in_x = in_x as usize;
                                let inp_idx = inp_offset + in_y * cont_s1 + in_x * cont_s2 + c_in;
                                let col_idx = patch_offset * tile_size + tile_idx;
                                col_tile[col_idx] = inp_cont[inp_idx];
                            }
                            // Move to next position (skip c_in channels)
                            patch_offset += p.c_in;
                        }
                    }
                }
            }

            // Now perform matmul: k_cache [c_out, k_size] @ col_tile [k_size, tile_size]
            let matmul = MatMul((1, p.c_out, tile_size, k_size));

            // Layouts for matmul
            // k_flat layout: [c_out, k_size] with stride [k_size, 1]
            // col_tile layout: [k_size, tile_size] with stride [tile_size, 1]
            let col_layout = Layout::contiguous((k_size, tile_size));

            // Perform matmul
            let result = matmul.f(&k_flat, &k_layout, &col_tile, &col_layout)?;

            // Copy results to output: result is [c_out, tile_size]
            for (tile_idx, (out_y, out_x)) in out_coords.iter().enumerate() {
                let dst_base = out_batch_offset + out_y * out_w + out_x;

                for c_out_idx in 0..p.c_out {
                    let dst_idx = dst_base + c_out_idx * (out_h * out_w);
                    let result_idx = c_out_idx * tile_size + tile_idx;
                    // SAFETY: Each batch processes a distinct region of the output buffer.
                    // Within each batch, tiles process non-overlapping output positions.
                    // Therefore, no two threads will write to the same dst_idx.
                    unsafe {
                        let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                        *ptr = result[result_idx];
                    }
                }
            }
            Ok::<(), crate::Error>(())
        })
    })?;

    Ok(dst)
}

/// General direct convolution impl. Decently fast for small inputs and kernels, but loses to full/tiled gemm.
fn conv2d_direct<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv2D,
    inp: &[T],
    inp_l: &Layout,
    k: &[T],
    k_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3) = crate::shape::dims4(inp_l.stride())?;
    let k = &k[k_l.start_offset()..];
    let (k_s0, k_s1, k_s2, k_s3) = crate::shape::dims4(k_l.stride())?;
    let (out_h, out_w) = (p.out_h(), p.out_w());

    // Output shape: [b_size, c_out, out_h, out_w].
    let mut dst = vec![T::zero(); p.b_size * p.c_out * out_h * out_w];

    // Pack NCHW input as NHWC for contiguous per-pixel channel dots. Do not
    // infer this from strides: NCHW can have the same numeric strides as NHWC
    // for shapes where channels equals width.
    let cont_s0 = p.i_h * p.i_w * p.c_in;
    let cont_s1 = p.i_w * p.c_in;
    let cont_s2 = p.c_in;
    let inp_cont: Vec<T> = {
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_h * p.i_w];
        for b_idx in 0..p.b_size {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    for c_idx in 0..p.c_in {
                        let src_idx =
                            b_idx * inp_s0 + c_idx * inp_s1 + h_idx * inp_s2 + w_idx * inp_s3;
                        let dst_idx = b_idx * cont_s0 + h_idx * cont_s1 + w_idx * cont_s2 + c_idx;
                        inp_cont[dst_idx] = inp[src_idx]
                    }
                }
            }
        }
        inp_cont
    };
    let inp_cont_len = inp_cont.len();

    let k_cache: Vec<Vec<T>> = (0..p.c_out)
        .map(|dst_c_idx| {
            (0..p.k_h * p.k_w)
                .flat_map(|kw_kh| {
                    let offset_h = kw_kh / p.k_w;
                    let offset_w = kw_kh % p.k_w;
                    (0..p.c_in).map(move |c_in_idx| {
                        k[dst_c_idx * k_s0 + c_in_idx * k_s1 + offset_h * k_s2 + offset_w * k_s3]
                    })
                })
                .collect()
        })
        .collect();

    for b_idx in 0..p.b_size {
        for offset_h in 0..p.k_h {
            for offset_w in 0..p.k_w {
                let k_offset = offset_h * p.k_w + offset_w;

                let batch_dst_start = b_idx * p.c_out * out_h * out_w;
                let batch_dst_end = batch_dst_start + p.c_out * out_h * out_w;
                dst[batch_dst_start..batch_dst_end]
                    .par_chunks_mut(out_h * out_w)
                    .enumerate()
                    .for_each(|(dst_c_idx, out_channel)| {
                    let k_cont = &k_cache[dst_c_idx][k_offset * p.c_in..(k_offset + 1) * p.c_in];
                    let batch_src_idx = b_idx * cont_s0;

                    for dst_h in 0..out_h {
                        let src_h = p.stride * dst_h + offset_h * p.dilation;
                        if src_h < p.padding || src_h >= p.i_h + p.padding {
                            continue;
                        }
                        let src_h = src_h - p.padding;
                        let h_src_idx = batch_src_idx + src_h * cont_s1;

                        for dst_w in 0..out_w {
                            let src_w = p.stride * dst_w + offset_w * p.dilation;
                            if src_w < p.padding || src_w >= p.i_w + p.padding {
                                continue;
                            }
                            let src_w = src_w - p.padding;
                            let dst_idx = dst_h * out_w + dst_w;
                            let inp_idx_1 = h_src_idx + src_w * cont_s2;
                            let inp_idx_2 = (inp_idx_1 + p.c_in).min(inp_cont_len);
                            let inp_cont = &inp_cont[inp_idx_1..inp_idx_2];
                            let mut d = T::zero();
                            unsafe {
                                T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in);
                            }
                            out_channel[dst_idx] += d;
                        }
                    }
                });
            }
        }
    }

    Ok(dst)
}

#[allow(clippy::uninit_vec)]
fn alloc_uninit_vec<T: WithDType + Copy + 'static>(size: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(size);
    unsafe { v.set_len(size) };
    v
}

/// Full im2col + gemm convolution implementation.
///
/// For large inputs im2col and copy_strided_src for output gets expensive.
fn conv2d_im2col_gemm<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv2D,
    inp: &[T],
    inp_l: &Layout,
    kernel: &[T],
    kernel_l: &Layout,
) -> Result<Vec<T>> {
    let op = Im2Col {
        h_k: p.k_h,
        w_k: p.k_w,
        padding: p.padding,
        stride: p.stride,
        dilation: p.dilation,
    };
    let col = op.f(inp, inp_l)?;
    let b = p.b_size;
    let n = p.c_out;
    let (h_out, w_out) = (p.out_h(), p.out_w());
    let k = op.h_k * op.w_k * p.c_in;
    let m = h_out * w_out;
    let col_l = Layout::contiguous((b, m, k));
    let res: Vec<T> = if kernel_l.is_contiguous() {
        let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
            .transpose(1, 2)?
            .broadcast_as((b, k, n))?;
        MatMul((b, m, n, k)).f(&col, &col_l, kernel, &kernel_l)?
    } else {
        // Make the kernel contiguous if not already the case.
        let mut kernel_c = alloc_uninit_vec(kernel_l.shape().elem_count());
        copy_strided_src_(kernel, &mut kernel_c, 0, kernel_l);
        let kernel_l = Layout::contiguous((1, n, k))
            .transpose(1, 2)?
            .broadcast_as((b, k, n))?;
        MatMul((b, m, n, k)).f(&col, &col_l, &kernel_c, &kernel_l)?
    };
    let res_l = Layout::contiguous((b, h_out, w_out, p.c_out))
        .transpose(1, 2)?
        .transpose(1, 3)?;
    let mut res_t = alloc_uninit_vec(res_l.shape().elem_count());
    copy_strided_src_(&res, &mut res_t, 0, &res_l);
    Ok(res_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max)
    }

    fn conv2d_reference(p: &ParamsConv2D, input: &[f32], kernel: &[f32]) -> Vec<f32> {
        let out_h = p.out_h();
        let out_w = p.out_w();
        let mut out = vec![0.0; p.b_size * p.c_out * out_h * out_w];
        for b in 0..p.b_size {
            for oc in 0..p.c_out {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;
                        for ic in 0..p.c_in {
                            for kh in 0..p.k_h {
                                let ih = oh * p.stride + kh * p.dilation;
                                if ih < p.padding || ih >= p.i_h + p.padding {
                                    continue;
                                }
                                let ih = ih - p.padding;
                                for kw in 0..p.k_w {
                                    let iw = ow * p.stride + kw * p.dilation;
                                    if iw < p.padding || iw >= p.i_w + p.padding {
                                        continue;
                                    }
                                    let iw = iw - p.padding;
                                    let input_idx =
                                        (((b * p.c_in + ic) * p.i_h + ih) * p.i_w) + iw;
                                    let kernel_idx =
                                        (((oc * p.c_in + ic) * p.k_h + kh) * p.k_w) + kw;
                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                        let out_idx = (((b * p.c_out + oc) * out_h + oh) * out_w) + ow;
                        out[out_idx] = sum;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn conv2d_impls_match_for_large_channels() {
        for (c_in, c_out, h, w) in [
            (32, 64, 16, 16),
            (64, 64, 16, 16),
            (64, 128, 8, 8),
            (128, 128, 8, 8),
            (128, 256, 4, 4),
            (256, 128, 4, 4),
            (256, 128, 8, 8),
            (128, 64, 8, 8),
            (128, 64, 16, 16),
            (64, 32, 16, 16),
            (64, 32, 32, 32),
            (32, 64, 32, 32),
            (32, 128, 32, 32),
        ] {
            assert_conv2d_impls_match(c_in, c_out, h, w);
        }
    }

    fn assert_conv2d_impls_match(c_in: usize, c_out: usize, h: usize, w: usize) {
        let params = ParamsConv2D {
            b_size: 1,
            i_h: h,
            i_w: w,
            k_h: 3,
            k_w: 3,
            c_out,
            c_in,
            padding: 1,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let input = (0..params.b_size * params.c_in * params.i_h * params.i_w)
            .map(|v| (v % 17) as f32 / 8.0 - 1.0)
            .collect::<Vec<_>>();
        let kernel = (0..params.c_out * params.c_in * params.k_h * params.k_w)
            .map(|v| (v % 13) as f32 / 6.0 - 1.0)
            .collect::<Vec<_>>();
        let input_l = Layout::contiguous((params.b_size, params.c_in, params.i_h, params.i_w));
        let kernel_l =
            Layout::contiguous((params.c_out, params.c_in, params.k_h, params.k_w));

        let full = conv2d_im2col_gemm(&params, &input, &input_l, &kernel, &kernel_l).unwrap();
        let direct = conv2d_direct(&params, &input, &input_l, &kernel, &kernel_l).unwrap();
        let tiled = conv2d_tiled(&params, &input, &input_l, &kernel, &kernel_l).unwrap();
        let reference = conv2d_reference(&params, &input, &kernel);

        let full_ref_max_abs = max_abs(&full, &reference);
        let direct_max_abs = max_abs(&direct, &full);
        let tiled_max_abs = max_abs(&tiled, &full);
        if direct_max_abs >= 1e-4 || tiled_max_abs >= 1e-4 {
            let out_hw = params.out_h() * params.out_w();
            if let Some((index, _)) = direct
                .iter()
                .zip(reference.iter())
                .enumerate()
                .find(|(_, (left, right))| (*left - *right).abs() >= 1e-4)
            {
                eprintln!(
                    "first direct mismatch index={index} oc={} pos={} direct={} reference={} full={}",
                    index / out_hw,
                    index % out_hw,
                    direct[index],
                    reference[index],
                    full[index]
                );
            }
        }
        assert!(
            full_ref_max_abs < 1e-4 && direct_max_abs < 1e-4 && tiled_max_abs < 1e-4,
            "c_in={c_in} c_out={c_out} h={h} w={w} full_ref_max_abs={full_ref_max_abs} direct_max_abs={direct_max_abs} tiled_max_abs={tiled_max_abs}"
        );
    }

    #[test]
    fn conv2d_impls_match_for_offset_layouts() {
        let params = ParamsConv2D {
            b_size: 1,
            i_h: 8,
            i_w: 8,
            k_h: 3,
            k_w: 3,
            c_out: 32,
            c_in: 64,
            padding: 1,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let input_prefix = 17;
        let kernel_prefix = 23;
        let input_len = params.b_size * params.c_in * params.i_h * params.i_w;
        let kernel_len = params.c_out * params.c_in * params.k_h * params.k_w;
        let mut input = vec![123.0; input_prefix];
        input.extend((0..input_len).map(|v| (v % 17) as f32 / 8.0 - 1.0));
        let mut kernel = vec![456.0; kernel_prefix];
        kernel.extend((0..kernel_len).map(|v| (v % 13) as f32 / 6.0 - 1.0));
        let input_l = Layout::contiguous_with_offset(
            (params.b_size, params.c_in, params.i_h, params.i_w),
            input_prefix,
        );
        let kernel_l = Layout::contiguous_with_offset(
            (params.c_out, params.c_in, params.k_h, params.k_w),
            kernel_prefix,
        );

        let full = conv2d_im2col_gemm(&params, &input, &input_l, &kernel, &kernel_l).unwrap();
        let direct = conv2d_direct(&params, &input, &input_l, &kernel, &kernel_l).unwrap();
        let tiled = conv2d_tiled(&params, &input, &input_l, &kernel, &kernel_l).unwrap();

        assert!(max_abs(&direct, &full) < 1e-4);
        assert!(max_abs(&tiled, &full) < 1e-4);
    }
}
