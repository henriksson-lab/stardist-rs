use std::ops::Range;

pub const OBJECT_KEYS: [&str; 6] = ["prob", "points", "coord", "dist", "class_prob", "class_id"];
pub const COORD_KEYS: [&str; 2] = ["points", "coord"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotFullyVisible {
    LargerThanBlockWriteRegion,
    LargerThanMinOverlap,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BigError {
    #[error("invalid block cover parameters")]
    InvalidCoverParameters,
    #[error("invalid block chain operation")]
    InvalidBlockChain,
    #[error("axes contain duplicate entries")]
    DuplicateAxis,
    #[error("axes length does not match dimensionality")]
    DimensionMismatch,
    #[error("axis is not present in block axes")]
    MissingAxis,
    #[error("query interval is outside the block write region")]
    InvalidBoundingBox,
    #[error("array length does not match shape")]
    ShapeMismatch,
    #[error("block slice dimensionality does not match array dimensionality")]
    SliceDimensionMismatch,
    #[error("coordinate input is empty")]
    EmptyCoordinates,
    #[error("distance length does not match number of rays")]
    RaysLengthMismatch,
    #[error("label image should be 2- or 3-dimensional")]
    WrongLabelDimension,
    #[error("object is not fully visible in this block")]
    NotFullyVisible(NotFullyVisible),
    #[error("polys dictionary must contain a coordinate key")]
    MissingCoordinateKey,
    #[error("polys object array shape does not match object count")]
    PolysShapeMismatch,
    #[error("polys coordinate array shape does not match block dimensionality")]
    PolysCoordinateShapeMismatch,
    #[error("This function has moved to {destination}.predict_instances_big.")]
    PredictBigMoved { destination: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub size: usize,
    pub min_overlap: usize,
    pub context: usize,
    pub pred: Option<usize>,
    pub succ: Option<usize>,
    pub stride: usize,
    pub start: usize,
    pub frozen: bool,
    pub extra_context_start: usize,
    pub extra_context_end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockND {
    pub id: usize,
    pub blocks: Vec<Block>,
    pub axes: Vec<char>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polygon;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polyhedron;

#[derive(Clone, Debug, PartialEq)]
pub enum BigPolysValue {
    F32 {
        values: Vec<f32>,
        shape: Vec<usize>,
    },
    I32 {
        values: Vec<i32>,
        shape: Vec<usize>,
    },
    Usize {
        values: Vec<usize>,
        shape: Vec<usize>,
    },
    Bool {
        values: Vec<bool>,
        shape: Vec<usize>,
    },
    Text(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct BigPolys {
    pub entries: Vec<(String, BigPolysValue)>,
}

pub fn _grid_divisible(grid: usize, size: usize) -> Result<usize, BigError> {
    if grid == 0 {
        return Err(BigError::InvalidCoverParameters);
    }
    if size % grid == 0 {
        Ok(size)
    } else {
        Ok(size.div_ceil(grid) * grid)
    }
}

pub fn predict_big(model_class_name: Option<&str>) -> Result<(), BigError> {
    let destination = match model_class_name {
        Some(name @ ("StarDist2D" | "StarDist3D")) => name.to_string(),
        _ => "{StarDist2D, StarDist3D}".to_string(),
    };
    Err(BigError::PredictBigMoved { destination })
}

fn axes_check_and_normalize(axes: &str, length: usize) -> Result<Vec<char>, BigError> {
    if axes.chars().count() != length {
        return Err(BigError::DimensionMismatch);
    }
    let mut out = Vec::with_capacity(length);
    for axis in axes.chars() {
        if out.contains(&axis) {
            return Err(BigError::DuplicateAxis);
        }
        out.push(axis);
    }
    Ok(out)
}

impl Polygon {
    pub fn coords_bbox(
        coords: &[&[[f32; 2]]],
        shape_max: Option<[usize; 2]>,
    ) -> Result<Vec<Range<usize>>, BigError> {
        if coords.is_empty() || coords.iter().any(|coord| coord.is_empty()) {
            return Err(BigError::EmptyCoordinates);
        }
        let mut mins = [f32::INFINITY; 2];
        let mut maxs = [f32::NEG_INFINITY; 2];
        for coord in coords {
            for point in *coord {
                mins[0] = mins[0].min(point[0]);
                mins[1] = mins[1].min(point[1]);
                maxs[0] = maxs[0].max(point[0]);
                maxs[1] = maxs[1].max(point[1]);
            }
        }
        let shape_max = shape_max.unwrap_or([usize::MAX, usize::MAX]);
        let min0 = mins[0].floor();
        let min1 = mins[1].floor();
        Ok(vec![
            (if min0 < 0.0 { 0 } else { min0 as usize })..shape_max[0].min(maxs[0].ceil() as usize),
            (if min1 < 0.0 { 0 } else { min1 as usize })..shape_max[1].min(maxs[1].ceil() as usize),
        ])
    }
}

impl Polyhedron {
    pub fn coords_bbox(
        dist_origin: &[(&[f32], [f32; 3])],
        rays: &crate::Rays,
        shape_max: Option<[usize; 3]>,
    ) -> Result<Vec<Range<usize>>, BigError> {
        if dist_origin.is_empty() {
            return Err(BigError::EmptyCoordinates);
        }
        let mut mins = [f32::INFINITY; 3];
        let mut maxs = [f32::NEG_INFINITY; 3];
        for (dist, point) in dist_origin {
            if dist.len() != rays.vertices.len() {
                return Err(BigError::RaysLengthMismatch);
            }
            for (d, vertex) in dist.iter().zip(&rays.vertices) {
                for axis in 0..3 {
                    let coord = d * vertex[axis] + point[axis];
                    mins[axis] = mins[axis].min(coord);
                    maxs[axis] = maxs[axis].max(coord);
                }
            }
        }
        let shape_max = shape_max.unwrap_or([usize::MAX, usize::MAX, usize::MAX]);
        let min0 = mins[0].floor();
        let min1 = mins[1].floor();
        let min2 = mins[2].floor();
        Ok(vec![
            (if min0 < 0.0 { 0 } else { min0 as usize })..shape_max[0].min(maxs[0].ceil() as usize),
            (if min1 < 0.0 { 0 } else { min1 as usize })..shape_max[1].min(maxs[1].ceil() as usize),
            (if min2 < 0.0 { 0 } else { min2 as usize })..shape_max[2].min(maxs[2].ceil() as usize),
        ])
    }
}

impl Block {
    pub fn start(&self, blocks: &[Block]) -> Result<usize, BigError> {
        if self.frozen || self.at_begin() {
            Ok(self.start)
        } else {
            let pred = self.pred.ok_or(BigError::InvalidBlockChain)?;
            let pred = blocks.get(pred).ok_or(BigError::InvalidBlockChain)?;
            Ok(pred.start(blocks)? + pred.stride)
        }
    }

    pub fn end(&self) -> usize {
        self.start + self.size
    }

    pub fn succ_start(&self) -> usize {
        self.start + self.stride
    }

    pub fn add_succ(blocks: &mut Vec<Block>, index: usize) -> Result<usize, BigError> {
        let block = blocks.get(index).ok_or(BigError::InvalidBlockChain)?;
        if block.succ.is_some() || block.frozen {
            return Err(BigError::InvalidBlockChain);
        }
        let succ = Block {
            size: block.size,
            min_overlap: block.min_overlap,
            context: block.context,
            pred: Some(index),
            succ: None,
            stride: block.size - (block.min_overlap + 2 * block.context),
            start: block.succ_start(),
            frozen: false,
            extra_context_start: 0,
            extra_context_end: 0,
        };
        let succ_index = blocks.len();
        blocks[index].succ = Some(succ_index);
        blocks.push(succ);
        Ok(succ_index)
    }

    pub fn decrease_stride(&mut self, amount: usize) -> Result<(), BigError> {
        if amount >= self.stride || self.frozen {
            return Err(BigError::InvalidBlockChain);
        }
        self.stride -= amount;
        Ok(())
    }

    pub fn freeze(blocks: &mut [Block], index: usize) -> Result<(), BigError> {
        let pred_frozen_or_begin = {
            let block = blocks.get(index).ok_or(BigError::InvalidBlockChain)?;
            !block.frozen
                && (block.at_begin()
                    || block
                        .pred
                        .and_then(|pred| blocks.get(pred))
                        .map(|pred| pred.frozen)
                        .unwrap_or(false))
        };
        if !pred_frozen_or_begin {
            return Err(BigError::InvalidBlockChain);
        }
        let start = if blocks[index].at_begin() {
            blocks[index].start
        } else {
            let pred = blocks[index].pred.ok_or(BigError::InvalidBlockChain)?;
            blocks
                .get(pred)
                .ok_or(BigError::InvalidBlockChain)?
                .succ_start()
        };
        blocks[index].start = start;
        blocks[index].frozen = true;
        if let Some(succ) = blocks[index].succ {
            Self::freeze(blocks, succ)?;
        }
        Ok(())
    }

    pub fn chain(blocks: &[Block], index: usize) -> Result<Vec<usize>, BigError> {
        let mut chain = Vec::new();
        let mut current = index;
        loop {
            let block = blocks.get(current).ok_or(BigError::InvalidBlockChain)?;
            chain.push(current);
            if let Some(succ) = block.succ {
                current = succ;
            } else {
                break;
            }
        }
        Ok(chain)
    }

    pub fn slice_read(&self) -> Range<usize> {
        self.start..self.end()
    }

    pub fn slice_crop_context(&self) -> Range<usize> {
        self.context_start()..(self.size - self.context_end())
    }

    pub fn slice_write(&self) -> Range<usize> {
        (self.start + self.context_start())..(self.end() - self.context_end())
    }

    pub fn is_responsible(&self, bbox: Range<usize>, blocks: &[Block]) -> Result<bool, BigError> {
        let bmin = bbox.start;
        let bmax = bbox.end;
        let r_start = if self.at_begin() {
            0
        } else {
            let pred = &blocks[self.pred.expect("non-begin block has predecessor")];
            pred.overlap() - pred.context_end() - self.context_start()
        };
        let r_end = self.size - self.context_start() - self.context_end();
        if !(bmin < bmax && bmax <= r_end) {
            return Err(BigError::InvalidBoundingBox);
        }

        if bmin == 0 && bmax >= r_start {
            if bmax == r_end {
                return Err(BigError::NotFullyVisible(
                    NotFullyVisible::LargerThanBlockWriteRegion,
                ));
            }
            if !self.at_begin() {
                return Err(BigError::NotFullyVisible(
                    NotFullyVisible::LargerThanMinOverlap,
                ));
            }
        }

        if bmax < r_start {
            return Ok(false);
        }
        if bmax == r_end && !self.at_end() {
            return Ok(false);
        }
        Ok(true)
    }

    pub fn at_begin(&self) -> bool {
        self.pred.is_none()
    }

    pub fn at_end(&self) -> bool {
        self.succ.is_none()
    }

    pub fn overlap(&self) -> usize {
        self.size - self.stride
    }

    pub fn context_start(&self) -> usize {
        if self.at_begin() {
            0
        } else {
            self.context + self.extra_context_start
        }
    }

    pub fn context_end(&self) -> usize {
        if self.at_end() {
            0
        } else {
            self.context + self.extra_context_end
        }
    }

    pub fn repr(&self, blocks: &[Block]) -> Result<String, BigError> {
        let start = self.start(blocks)?;
        let end = start + self.size;
        let context_start = self.context_start();
        let context_end = self.context_end();
        let write_start = start + context_start;
        let write_end = end - context_end;
        let mut text = format!(
            "Block({start:03}:{end:03}, write={write_start:03}:{write_end:03}, size={context_start}+{}+{context_end}",
            self.size - context_start - context_end
        );
        if let Some(succ) = self.succ {
            let succ = blocks.get(succ).ok_or(BigError::InvalidBlockChain)?;
            text.push_str(&format!(
                ", overlap={}R/{}W",
                self.overlap(),
                self.overlap() - self.context_end() - succ.context_start()
            ));
        }
        text.push(')');
        Ok(text)
    }

    pub fn cover(
        size: usize,
        block_size: usize,
        min_overlap: usize,
        context: usize,
        grid: usize,
    ) -> Result<Vec<Block>, BigError> {
        if !(min_overlap + 2 * context < block_size
            && block_size <= size
            && 0 < grid
            && grid <= block_size)
        {
            return Err(BigError::InvalidCoverParameters);
        }
        let block_size = _grid_divisible(grid, block_size)?;
        let min_overlap = _grid_divisible(grid, min_overlap)?;
        let context = _grid_divisible(grid, context)?;
        let size_orig = size;
        let size = _grid_divisible(grid, size)?;
        if !(block_size <= size && min_overlap + 2 * context < block_size) {
            return Err(BigError::InvalidCoverParameters);
        }

        let size_grid = size / grid;
        let block_size_grid = block_size / grid;
        let min_overlap_grid = min_overlap / grid;
        let context_grid = context / grid;

        let mut blocks = vec![Block {
            size: block_size_grid,
            min_overlap: min_overlap_grid,
            context: context_grid,
            pred: None,
            succ: None,
            stride: block_size_grid - (min_overlap_grid + 2 * context_grid),
            start: 0,
            frozen: false,
            extra_context_start: 0,
            extra_context_end: 0,
        }];
        while blocks.last().expect("block cover is non-empty").end() < size_grid {
            let pred = blocks.len() - 1;
            let stride = block_size_grid - (min_overlap_grid + 2 * context_grid);
            let start = blocks[pred].succ_start();
            blocks[pred].succ = Some(pred + 1);
            blocks.push(Block {
                size: block_size_grid,
                min_overlap: min_overlap_grid,
                context: context_grid,
                pred: Some(pred),
                succ: None,
                stride,
                start,
                frozen: false,
                extra_context_start: 0,
                extra_context_end: 0,
            });
        }

        let last = blocks.len() - 1;
        let mut excess = blocks[last].end() - size_grid;
        let mut t = 0usize;
        while excess > 0 {
            if blocks[t].stride == 0 {
                return Err(BigError::InvalidCoverParameters);
            }
            blocks[t].stride -= 1;
            excess -= 1;
            t += 1;
            if t == last {
                t = 0;
            }
        }
        for i in 1..blocks.len() {
            blocks[i].start = blocks[i - 1].succ_start();
        }

        if blocks.len() >= 3 {
            for i in 0..blocks.len() - 2 {
                let overlap_write = blocks[i]
                    .slice_write()
                    .end
                    .saturating_sub(blocks[i + 2].slice_write().start);
                if overlap_write > 0 {
                    let overlap_split1 = overlap_write / 2;
                    let overlap_split2 = overlap_write - overlap_split1;
                    blocks[i].extra_context_end += overlap_split1;
                    blocks[i + 2].extra_context_start += overlap_split2;
                }
            }
        }

        if grid > 1 {
            for block in &mut blocks {
                block.size *= grid;
                block.min_overlap *= grid;
                block.context *= grid;
                block.stride *= grid;
                block.start *= grid;
                block.extra_context_start *= grid;
                block.extra_context_end *= grid;
            }
            let size_delta = size - size_orig;
            let last = blocks.len() - 1;
            blocks[last].size -= size_delta;
        }
        for block in &mut blocks {
            block.frozen = true;
        }

        let last = blocks.last().expect("block cover is non-empty");
        if blocks.first().expect("block cover is non-empty").start != 0 || last.end() != size_orig {
            return Err(BigError::InvalidCoverParameters);
        }
        for i in 0..blocks.len().saturating_sub(1) {
            if blocks[i].overlap() < 2 * context || blocks[i].overlap() - 2 * context < min_overlap
            {
                return Err(BigError::InvalidCoverParameters);
            }
            if blocks[i].slice_write().end - blocks[i + 1].slice_write().start < min_overlap {
                return Err(BigError::InvalidCoverParameters);
            }
            if blocks[i].start % grid != 0 || blocks[i].end() % grid != 0 {
                return Err(BigError::InvalidCoverParameters);
            }
        }
        if blocks.len() >= 3 {
            for i in 0..blocks.len() - 2 {
                if blocks[i].slice_write().end > blocks[i + 2].slice_write().start {
                    return Err(BigError::InvalidCoverParameters);
                }
            }
        }

        Ok(blocks)
    }
}

impl BlockND {
    pub fn repr(&self) -> String {
        let slices = self
            .blocks
            .iter()
            .zip(&self.axes)
            .map(|(block, axis)| format!("{axis}={:03}:{:03}", block.start, block.end()))
            .collect::<Vec<_>>()
            .join(",");
        format!("BlockND({}|{slices})", self.id)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Block> {
        self.blocks.iter()
    }

    pub fn blocks_for_axes(&self, axes: Option<&str>) -> Result<Vec<&Block>, BigError> {
        let axes = match axes {
            Some(axes) => axes_check_and_normalize(axes, axes.chars().count())?,
            None => self.axes.clone(),
        };
        let mut blocks = Vec::with_capacity(axes.len());
        for axis in axes {
            let Some(index) = self.axes.iter().position(|a| *a == axis) else {
                return Err(BigError::MissingAxis);
            };
            blocks.push(&self.blocks[index]);
        }
        Ok(blocks)
    }

    pub fn slice_read(&self, axes: Option<&str>) -> Result<Vec<Range<usize>>, BigError> {
        Ok(self
            .blocks_for_axes(axes)?
            .into_iter()
            .map(Block::slice_read)
            .collect())
    }

    pub fn slice_crop_context(&self, axes: Option<&str>) -> Result<Vec<Range<usize>>, BigError> {
        Ok(self
            .blocks_for_axes(axes)?
            .into_iter()
            .map(Block::slice_crop_context)
            .collect())
    }

    pub fn slice_write(&self, axes: Option<&str>) -> Result<Vec<Range<usize>>, BigError> {
        Ok(self
            .blocks_for_axes(axes)?
            .into_iter()
            .map(Block::slice_write)
            .collect())
    }

    pub fn read<T: Clone>(
        &self,
        x: &[T],
        shape: &[usize],
        axes: Option<&str>,
    ) -> Result<(Vec<T>, Vec<usize>), BigError> {
        let slices = self.slice_read(axes)?;
        if slices.len() != shape.len() {
            return Err(BigError::SliceDimensionMismatch);
        }
        if x.len() != shape.iter().product::<usize>() {
            return Err(BigError::ShapeMismatch);
        }
        for (slice, dim) in slices.iter().zip(shape) {
            if slice.start > slice.end || slice.end > *dim {
                return Err(BigError::ShapeMismatch);
            }
        }

        let out_shape = slices
            .iter()
            .map(|slice| slice.end - slice.start)
            .collect::<Vec<_>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        let mut out = Vec::with_capacity(out_len);
        for out_index in 0..out_len {
            let mut rem = out_index;
            let mut src_index = 0usize;
            for axis in (0..shape.len()).rev() {
                let len = out_shape[axis];
                let coord = rem % len;
                rem /= len;
                src_index += (slices[axis].start + coord) * strides[axis];
            }
            out.push(x[src_index].clone());
        }
        Ok((out, out_shape))
    }

    pub fn crop_context<T: Clone>(
        &self,
        labels: &[T],
        shape: &[usize],
        axes: Option<&str>,
    ) -> Result<(Vec<T>, Vec<usize>), BigError> {
        let slices = self.slice_crop_context(axes)?;
        if slices.len() != shape.len() {
            return Err(BigError::SliceDimensionMismatch);
        }
        if labels.len() != shape.iter().product::<usize>() {
            return Err(BigError::ShapeMismatch);
        }
        for (slice, dim) in slices.iter().zip(shape) {
            if slice.start > slice.end || slice.end > *dim {
                return Err(BigError::ShapeMismatch);
            }
        }

        let out_shape = slices
            .iter()
            .map(|slice| slice.end - slice.start)
            .collect::<Vec<_>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        let mut out = Vec::with_capacity(out_len);
        for out_index in 0..out_len {
            let mut rem = out_index;
            let mut src_index = 0usize;
            for axis in (0..shape.len()).rev() {
                let len = out_shape[axis];
                let coord = rem % len;
                rem /= len;
                src_index += (slices[axis].start + coord) * strides[axis];
            }
            out.push(labels[src_index].clone());
        }
        Ok((out, out_shape))
    }

    pub fn write(
        &self,
        x: &mut [i32],
        shape: &[usize],
        labels: &[i32],
        labels_shape: &[usize],
        axes: Option<&str>,
    ) -> Result<(), BigError> {
        let slices = self.slice_write(axes)?;
        if slices.len() != shape.len() || slices.len() != labels_shape.len() {
            return Err(BigError::SliceDimensionMismatch);
        }
        if x.len() != shape.iter().product::<usize>()
            || labels.len() != labels_shape.iter().product::<usize>()
        {
            return Err(BigError::ShapeMismatch);
        }
        for ((slice, dim), label_dim) in slices.iter().zip(shape).zip(labels_shape) {
            if slice.start > slice.end || slice.end > *dim || slice.end - slice.start != *label_dim
            {
                return Err(BigError::ShapeMismatch);
            }
        }

        let mut x_strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            x_strides[i] = x_strides[i + 1] * shape[i + 1];
        }
        let mut label_strides = vec![1usize; labels_shape.len()];
        for i in (0..labels_shape.len().saturating_sub(1)).rev() {
            label_strides[i] = label_strides[i + 1] * labels_shape[i + 1];
        }

        for label_index in 0..labels.len() {
            if labels[label_index] <= 0 {
                continue;
            }
            let mut rem = label_index;
            let mut dst_index = 0usize;
            for axis in (0..labels_shape.len()).rev() {
                let coord = rem % labels_shape[axis];
                rem /= labels_shape[axis];
                dst_index += (slices[axis].start + coord) * x_strides[axis];
            }
            x[dst_index] = labels[label_index];
        }
        Ok(())
    }

    pub fn is_responsible(
        &self,
        slices: &[Range<usize>],
        axes: Option<&str>,
    ) -> Result<bool, BigError> {
        let blocks = self.blocks_for_axes(axes)?;
        if blocks.len() != slices.len() {
            return Err(BigError::DimensionMismatch);
        }
        for (block, slice) in blocks.iter().zip(slices) {
            if !block.is_responsible(slice.clone(), &self.blocks)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn filter_objects(
        &self,
        labels: &[i32],
        shape: &[usize],
        axes: Option<&str>,
    ) -> Result<Vec<i32>, BigError> {
        let ndim = self.blocks_for_axes(axes)?.len();
        if ndim != 2 && ndim != 3 {
            return Err(BigError::WrongLabelDimension);
        }
        if labels.len() != shape.iter().product::<usize>() {
            return Err(BigError::ShapeMismatch);
        }
        if shape.len() != ndim {
            return Err(BigError::DimensionMismatch);
        }
        let crop_shape = self
            .slice_crop_context(axes)?
            .iter()
            .map(|slice| slice.end - slice.start)
            .collect::<Vec<_>>();
        if crop_shape != shape {
            return Err(BigError::ShapeMismatch);
        }

        let mut labels_sorted = labels
            .iter()
            .copied()
            .filter(|label| *label > 0)
            .collect::<Vec<_>>();
        labels_sorted.sort_unstable();
        labels_sorted.dedup();
        let mut labels_filtered = vec![0i32; labels.len()];
        for label in labels_sorted {
            let mut mins = vec![usize::MAX; ndim];
            let mut maxs = vec![0usize; ndim];
            let mut found = false;
            for (index, value) in labels.iter().enumerate() {
                if *value != label {
                    continue;
                }
                found = true;
                let mut rem = index;
                let mut coord = vec![0usize; ndim];
                for axis in (0..ndim).rev() {
                    coord[axis] = rem % shape[axis];
                    rem /= shape[axis];
                }
                for axis in 0..ndim {
                    mins[axis] = mins[axis].min(coord[axis]);
                    maxs[axis] = maxs[axis].max(coord[axis] + 1);
                }
            }
            if !found {
                continue;
            }
            let slices = mins
                .iter()
                .zip(&maxs)
                .map(|(start, end)| *start..*end)
                .collect::<Vec<_>>();
            if self.is_responsible(&slices, axes)? {
                if ndim == 2 {
                    let w = shape[1];
                    for y in slices[0].clone() {
                        for x in slices[1].clone() {
                            let index = y * w + x;
                            if labels[index] == label {
                                labels_filtered[index] = label;
                            }
                        }
                    }
                } else {
                    let h = shape[1];
                    let w = shape[2];
                    for z in slices[0].clone() {
                        for y in slices[1].clone() {
                            for x in slices[2].clone() {
                                let index = (z * h + y) * w + x;
                                if labels[index] == label {
                                    labels_filtered[index] = label;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(labels_filtered)
    }

    pub fn filter_objects_with_polys(
        &self,
        labels: &[i32],
        shape: &[usize],
        polys: &BigPolys,
        axes: Option<&str>,
    ) -> Result<(Vec<i32>, BigPolys), BigError> {
        if !polys
            .entries
            .iter()
            .any(|(key, _)| COORD_KEYS.contains(&key.as_str()))
        {
            return Err(BigError::MissingCoordinateKey);
        }
        let labels_filtered = self.filter_objects(labels, shape, axes)?;
        let mut filtered_labels = labels_filtered
            .iter()
            .copied()
            .filter(|label| *label > 0)
            .collect::<Vec<_>>();
        filtered_labels.sort_unstable();
        filtered_labels.dedup();
        let filtered_ind = filtered_labels
            .iter()
            .map(|label| (*label - 1) as usize)
            .collect::<Vec<_>>();
        let blocks = self.blocks_for_axes(axes)?;
        let ndim = blocks.len();
        let starts = blocks
            .iter()
            .map(|block| block.slice_read().start as f32)
            .collect::<Vec<_>>();

        let mut entries = Vec::<(String, BigPolysValue)>::with_capacity(polys.entries.len());
        for (key, value) in &polys.entries {
            let is_object_key = OBJECT_KEYS.contains(&key.as_str());
            let is_coord_key = COORD_KEYS.contains(&key.as_str());
            let mut value_out = match value {
                BigPolysValue::F32 { values, shape } if is_object_key => {
                    if shape.is_empty() || values.len() != shape.iter().product::<usize>() {
                        return Err(BigError::PolysShapeMismatch);
                    }
                    let slab = shape[1..].iter().product::<usize>();
                    let mut filtered = Vec::<f32>::with_capacity(filtered_ind.len() * slab);
                    for index in &filtered_ind {
                        if *index >= shape[0] {
                            return Err(BigError::PolysShapeMismatch);
                        }
                        filtered.extend_from_slice(&values[index * slab..(index + 1) * slab]);
                    }
                    let mut shape_out = shape.clone();
                    shape_out[0] = filtered_ind.len();
                    BigPolysValue::F32 {
                        values: filtered,
                        shape: shape_out,
                    }
                }
                BigPolysValue::I32 { values, shape } if is_object_key => {
                    if shape.is_empty() || values.len() != shape.iter().product::<usize>() {
                        return Err(BigError::PolysShapeMismatch);
                    }
                    let slab = shape[1..].iter().product::<usize>();
                    let mut filtered = Vec::<i32>::with_capacity(filtered_ind.len() * slab);
                    for index in &filtered_ind {
                        if *index >= shape[0] {
                            return Err(BigError::PolysShapeMismatch);
                        }
                        filtered.extend_from_slice(&values[index * slab..(index + 1) * slab]);
                    }
                    let mut shape_out = shape.clone();
                    shape_out[0] = filtered_ind.len();
                    BigPolysValue::I32 {
                        values: filtered,
                        shape: shape_out,
                    }
                }
                BigPolysValue::Usize { values, shape } if is_object_key => {
                    if shape.is_empty() || values.len() != shape.iter().product::<usize>() {
                        return Err(BigError::PolysShapeMismatch);
                    }
                    let slab = shape[1..].iter().product::<usize>();
                    let mut filtered = Vec::<usize>::with_capacity(filtered_ind.len() * slab);
                    for index in &filtered_ind {
                        if *index >= shape[0] {
                            return Err(BigError::PolysShapeMismatch);
                        }
                        filtered.extend_from_slice(&values[index * slab..(index + 1) * slab]);
                    }
                    let mut shape_out = shape.clone();
                    shape_out[0] = filtered_ind.len();
                    BigPolysValue::Usize {
                        values: filtered,
                        shape: shape_out,
                    }
                }
                BigPolysValue::Bool { values, shape } if is_object_key => {
                    if shape.is_empty() || values.len() != shape.iter().product::<usize>() {
                        return Err(BigError::PolysShapeMismatch);
                    }
                    let slab = shape[1..].iter().product::<usize>();
                    let mut filtered = Vec::<bool>::with_capacity(filtered_ind.len() * slab);
                    for index in &filtered_ind {
                        if *index >= shape[0] {
                            return Err(BigError::PolysShapeMismatch);
                        }
                        filtered.extend_from_slice(&values[index * slab..(index + 1) * slab]);
                    }
                    let mut shape_out = shape.clone();
                    shape_out[0] = filtered_ind.len();
                    BigPolysValue::Bool {
                        values: filtered,
                        shape: shape_out,
                    }
                }
                _ => value.clone(),
            };
            if is_coord_key {
                if let BigPolysValue::F32 { values, shape } = &mut value_out {
                    if shape.len() < 2
                        || shape[1] != ndim
                        || values.len() != shape.iter().product::<usize>()
                    {
                        return Err(BigError::PolysCoordinateShapeMismatch);
                    }
                    let stride_axis = shape[2..].iter().product::<usize>();
                    for (flat, value) in values.iter_mut().enumerate() {
                        let axis = (flat / stride_axis) % shape[1];
                        *value += starts[axis];
                    }
                } else {
                    return Err(BigError::PolysCoordinateShapeMismatch);
                }
            }
            entries.push((key.clone(), value_out));
        }
        Ok((labels_filtered, BigPolys { entries }))
    }

    pub fn translate_coordinates(
        &self,
        coordinates: &[Vec<f32>],
        axes: Option<&str>,
    ) -> Result<Vec<Vec<f32>>, BigError> {
        let blocks = self.blocks_for_axes(axes)?;
        let ndim = blocks.len();
        let mut translated = Vec::with_capacity(coordinates.len());
        for point in coordinates {
            if point.len() != ndim {
                return Err(BigError::DimensionMismatch);
            }
            translated.push(
                point
                    .iter()
                    .zip(&blocks)
                    .map(|(coord, block)| coord + block.slice_read().start as f32)
                    .collect(),
            );
        }
        Ok(translated)
    }

    pub fn cover(
        shape: &[usize],
        axes: &str,
        block_size: &[usize],
        min_overlap: &[usize],
        context: &[usize],
        grid: &[usize],
    ) -> Result<Vec<BlockND>, BigError> {
        let n = shape.len();
        let axes = axes_check_and_normalize(axes, n)?;
        if block_size.len() != n || min_overlap.len() != n || context.len() != n || grid.len() != n
        {
            return Err(BigError::DimensionMismatch);
        }

        let cover_1d = shape
            .iter()
            .zip(block_size)
            .zip(min_overlap)
            .zip(context)
            .zip(grid)
            .map(|((((size, block_size), min_overlap), context), grid)| {
                Block::cover(*size, *block_size, *min_overlap, *context, *grid)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut blocks_nd = Vec::new();
        let mut current = Vec::<Block>::with_capacity(n);
        fn product(
            axis: usize,
            axes: &[char],
            cover_1d: &[Vec<Block>],
            current: &mut Vec<Block>,
            blocks_nd: &mut Vec<BlockND>,
        ) {
            if axis == cover_1d.len() {
                blocks_nd.push(BlockND {
                    id: blocks_nd.len(),
                    blocks: current.clone(),
                    axes: axes.to_vec(),
                });
                return;
            }
            for block in &cover_1d[axis] {
                current.push(block.clone());
                product(axis + 1, axes, cover_1d, current, blocks_nd);
                current.pop();
            }
        }
        product(0, &axes, &cover_1d, &mut current, &mut blocks_nd);
        Ok(blocks_nd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_divisible_rounds_up_to_grid_multiple() {
        assert_eq!(_grid_divisible(8, 32).unwrap(), 32);
        assert_eq!(_grid_divisible(8, 33).unwrap(), 40);
        assert_eq!(_grid_divisible(1, 33).unwrap(), 33);
    }

    #[test]
    fn predict_big_reports_moved_destination_for_stardist_model() {
        assert_eq!(
            predict_big(Some("StarDist2D")),
            Err(BigError::PredictBigMoved {
                destination: "StarDist2D".to_string(),
            })
        );
        assert_eq!(
            predict_big(Some("StarDist3D")).unwrap_err().to_string(),
            "This function has moved to StarDist3D.predict_instances_big."
        );
    }

    #[test]
    fn predict_big_reports_generic_destination_for_unknown_model() {
        assert_eq!(
            predict_big(Some("OtherModel")),
            Err(BigError::PredictBigMoved {
                destination: "{StarDist2D, StarDist3D}".to_string(),
            })
        );
        assert_eq!(
            predict_big(None).unwrap_err().to_string(),
            "This function has moved to {StarDist2D, StarDist3D}.predict_instances_big."
        );
    }

    #[test]
    fn block_cover_matches_stardist_grid_aligned_layout() {
        let blocks = Block::cover(100, 32, 8, 4, 4).unwrap();
        let read: Vec<_> = blocks.iter().map(Block::slice_read).collect();
        let write: Vec<_> = blocks.iter().map(Block::slice_write).collect();
        assert_eq!(read, vec![0..32, 12..44, 24..56, 36..68, 52..84, 68..100]);
        assert_eq!(write, vec![0..28, 16..40, 28..52, 40..64, 56..80, 72..100]);
        assert!(blocks.iter().all(|b| b.frozen));
    }

    #[test]
    fn block_cover_keeps_only_neighboring_write_regions_overlapping() {
        let blocks = Block::cover(96, 48, 16, 4, 1).unwrap();
        let write: Vec<_> = blocks.iter().map(Block::slice_write).collect();
        assert_eq!(write, vec![0..44, 28..68, 52..96]);
        assert!(
            blocks
                .windows(3)
                .all(|w| w[0].slice_write().end <= w[2].slice_write().start)
        );
    }

    #[test]
    fn block_chain_methods_match_python_mutation_and_freeze_semantics() {
        let mut blocks = vec![Block {
            size: 32,
            min_overlap: 4,
            context: 2,
            pred: None,
            succ: None,
            stride: 24,
            start: 0,
            frozen: false,
            extra_context_start: 0,
            extra_context_end: 0,
        }];

        let succ = Block::add_succ(&mut blocks, 0).unwrap();
        assert_eq!(succ, 1);
        let succ2 = Block::add_succ(&mut blocks, 1).unwrap();
        assert_eq!(succ2, 2);
        assert_eq!(blocks[0].succ, Some(1));
        assert_eq!(blocks[1].pred, Some(0));
        assert_eq!(blocks[1].succ, Some(2));
        assert_eq!(blocks[2].pred, Some(1));
        assert_eq!(Block::chain(&blocks, 0).unwrap(), vec![0, 1, 2]);

        blocks[0].decrease_stride(4).unwrap();
        assert_eq!(blocks[0].succ_start(), 20);
        assert_eq!(blocks[1].start(&blocks).unwrap(), 20);
        assert_eq!(blocks[2].start(&blocks).unwrap(), 44);
        assert!(blocks[0].decrease_stride(20).is_err());
        assert!(Block::freeze(&mut blocks, 1).is_err());

        Block::freeze(&mut blocks, 0).unwrap();
        assert!(blocks.iter().all(|block| block.frozen));
        assert_eq!(blocks[1].start, 20);
        assert_eq!(blocks[1].slice_read(), 20..52);
        assert_eq!(blocks[2].start, 44);
        assert_eq!(blocks[2].slice_read(), 44..76);
        assert!(Block::add_succ(&mut blocks, 2).is_err());
    }

    #[test]
    fn block_repr_matches_python_debug_string_shape() {
        let blocks = Block::cover(100, 32, 8, 4, 4).unwrap();
        assert_eq!(
            blocks[0].repr(&blocks).unwrap(),
            "Block(000:032, write=000:028, size=0+28+4, overlap=20R/12W)"
        );
        assert_eq!(
            blocks[5].repr(&blocks).unwrap(),
            "Block(068:100, write=072:100, size=4+28+0)"
        );
    }

    #[test]
    fn block_responsibility_matches_overlap_rules() {
        let blocks = Block::cover(100, 32, 8, 4, 4).unwrap();
        assert!(blocks[0].is_responsible(0..8, &blocks).unwrap());
        assert!(!blocks[1].is_responsible(0..7, &blocks).unwrap());
        assert!(matches!(
            blocks[1].is_responsible(0..12, &blocks),
            Err(BigError::NotFullyVisible(
                NotFullyVisible::LargerThanMinOverlap
            ))
        ));
    }

    #[test]
    fn block_nd_cover_uses_cartesian_product_and_axis_order() {
        let blocks = BlockND::cover(&[64, 48], "YX", &[32, 24], &[8, 8], &[4, 4], &[4, 4]).unwrap();
        assert_eq!(blocks.len(), 12);
        assert_eq!(blocks[0].id, 0);
        assert_eq!(blocks[0].repr(), "BlockND(0|Y=000:032,X=000:024)");
        assert_eq!(
            blocks[0].iter().map(Block::slice_read).collect::<Vec<_>>(),
            vec![0..32, 0..24]
        );
        assert_eq!(
            blocks[0].slice_read(Some("XY")).unwrap(),
            vec![0..24, 0..32]
        );
        assert_eq!(blocks[11].slice_write(None).unwrap(), vec![36..64, 28..48]);
        assert_eq!(
            blocks[11]
                .translate_coordinates(&[vec![1.5, 2.0]], Some("YX"))
                .unwrap(),
            vec![vec![33.5, 26.0]]
        );
    }

    #[test]
    fn block_nd_read_crop_context_and_write_match_numpy_slicing_semantics() {
        let block =
            BlockND::cover(&[8, 8], "YX", &[6, 6], &[2, 2], &[1, 1], &[1, 1]).unwrap()[0].clone();
        let image = (0..64).collect::<Vec<i32>>();

        let (tile, tile_shape) = block.read(&image, &[8, 8], Some("YX")).unwrap();
        assert_eq!(tile_shape, vec![6, 6]);
        assert_eq!(
            tile,
            vec![
                0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28,
                29, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45,
            ]
        );

        let (cropped, cropped_shape) = block.crop_context(&tile, &tile_shape, None).unwrap();
        assert_eq!(cropped_shape, vec![5, 5]);
        assert_eq!(
            cropped,
            vec![
                0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 32, 33,
                34, 35, 36,
            ]
        );

        let mut labels_out = vec![0i32; 64];
        let mut labels = vec![0i32; 25];
        labels[0] = 4;
        labels[6] = 7;
        labels[24] = 9;
        block
            .write(&mut labels_out, &[8, 8], &labels, &[5, 5], Some("YX"))
            .unwrap();
        assert_eq!(labels_out[0], 4);
        assert_eq!(labels_out[9], 7);
        assert_eq!(labels_out[36], 9);
        assert_eq!(labels_out.iter().filter(|&&v| v > 0).count(), 3);
    }

    #[test]
    fn polygon_coords_bbox_matches_floor_ceil_and_shape_clipping() {
        let coord_a = vec![[-1.2, 2.1], [4.8, 6.2]];
        let coord_b = vec![[3.4, -2.0]];
        assert_eq!(
            Polygon::coords_bbox(&[&coord_a, &coord_b], Some([5, 5])).unwrap(),
            vec![0..5, 0..5]
        );

        let coord_c = vec![[1.2, 2.2], [3.7, 4.1]];
        assert_eq!(
            Polygon::coords_bbox(&[&coord_c], None).unwrap(),
            vec![1..4, 2..5]
        );
    }

    #[test]
    fn polyhedron_coords_bbox_matches_ray_vertices_and_shape_clipping() {
        let rays = crate::Rays {
            name: "test".to_string(),
            kwargs: crate::RaysKwargs::default(),
            vertices: vec![
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            faces: Vec::new(),
        };
        let dist = vec![2.0, 3.0, 4.0, 5.0];
        assert_eq!(
            Polyhedron::coords_bbox(&[(&dist, [10.0, 10.0, 10.0])], &rays, Some([11, 13, 20]))
                .unwrap(),
            vec![7..11, 10..13, 10..15]
        );

        assert_eq!(
            Polyhedron::coords_bbox(&[(&dist[..3], [10.0, 10.0, 10.0])], &rays, None),
            Err(BigError::RaysLengthMismatch)
        );
    }

    #[test]
    fn block_nd_filter_objects_keeps_only_responsible_2d_regions() {
        let block =
            BlockND::cover(&[8, 8], "YX", &[6, 6], &[2, 2], &[1, 1], &[1, 1]).unwrap()[1].clone();
        assert_eq!(block.slice_crop_context(None).unwrap(), vec![0..5, 1..6]);

        let mut labels = vec![0i32; 25];
        labels[0] = 1;
        labels[2] = 2;
        labels[7] = 2;
        labels[24] = 3;
        let filtered = block.filter_objects(&labels, &[5, 5], None).unwrap();
        assert_eq!(filtered[0], 0);
        assert_eq!(filtered[1], 0);
        assert_eq!(filtered[2], 2);
        assert_eq!(filtered[7], 2);
        assert_eq!(filtered[24], 0);
    }

    #[test]
    fn block_nd_filter_objects_with_polys_filters_object_arrays_and_translates_coordinates() {
        let block =
            BlockND::cover(&[8, 8], "YX", &[6, 6], &[2, 2], &[1, 1], &[1, 1]).unwrap()[1].clone();
        assert_eq!(block.slice_read(None).unwrap(), vec![0..6, 2..8]);

        let mut labels = vec![0i32; 25];
        labels[0] = 1;
        labels[2] = 2;
        labels[7] = 2;
        labels[24] = 3;
        let polys = BigPolys {
            entries: vec![
                (
                    "prob".to_string(),
                    BigPolysValue::F32 {
                        values: vec![0.1, 0.9, 0.3],
                        shape: vec![3],
                    },
                ),
                (
                    "points".to_string(),
                    BigPolysValue::F32 {
                        values: vec![0.0, 0.0, 2.0, 3.0, 4.0, 4.0],
                        shape: vec![3, 2],
                    },
                ),
                (
                    "coord".to_string(),
                    BigPolysValue::F32 {
                        values: vec![
                            0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 22.0, 23.0, 40.0, 41.0, 42.0, 43.0,
                        ],
                        shape: vec![3, 2, 2],
                    },
                ),
                (
                    "dist".to_string(),
                    BigPolysValue::F32 {
                        values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        shape: vec![3, 2],
                    },
                ),
                (
                    "class_id".to_string(),
                    BigPolysValue::Usize {
                        values: vec![2, 1, 0],
                        shape: vec![3],
                    },
                ),
                (
                    "rays_json".to_string(),
                    BigPolysValue::Text("unchanged".to_string()),
                ),
            ],
        };

        let (filtered_labels, filtered_polys) = block
            .filter_objects_with_polys(&labels, &[5, 5], &polys, None)
            .unwrap();
        assert_eq!(filtered_labels[2], 2);
        assert_eq!(filtered_labels[7], 2);
        assert_eq!(
            filtered_labels.iter().filter(|&&label| label > 0).count(),
            2
        );
        assert_eq!(
            filtered_polys.entries,
            vec![
                (
                    "prob".to_string(),
                    BigPolysValue::F32 {
                        values: vec![0.9],
                        shape: vec![1],
                    },
                ),
                (
                    "points".to_string(),
                    BigPolysValue::F32 {
                        values: vec![2.0, 5.0],
                        shape: vec![1, 2],
                    },
                ),
                (
                    "coord".to_string(),
                    BigPolysValue::F32 {
                        values: vec![20.0, 21.0, 24.0, 25.0],
                        shape: vec![1, 2, 2],
                    },
                ),
                (
                    "dist".to_string(),
                    BigPolysValue::F32 {
                        values: vec![3.0, 4.0],
                        shape: vec![1, 2],
                    },
                ),
                (
                    "class_id".to_string(),
                    BigPolysValue::Usize {
                        values: vec![1],
                        shape: vec![1],
                    },
                ),
                (
                    "rays_json".to_string(),
                    BigPolysValue::Text("unchanged".to_string()),
                ),
            ]
        );
    }

    #[test]
    fn block_nd_filter_objects_keeps_3d_regions() {
        let block = BlockND::cover(
            &[8, 8, 8],
            "ZYX",
            &[6, 6, 6],
            &[2, 2, 2],
            &[1, 1, 1],
            &[1, 1, 1],
        )
        .unwrap()[0]
            .clone();
        assert_eq!(
            block.slice_crop_context(None).unwrap(),
            vec![0..5, 0..5, 0..5]
        );

        let mut labels = vec![0i32; 125];
        labels[(2 * 5 + 2) * 5 + 2] = 4;
        labels[(2 * 5 + 2) * 5 + 3] = 4;
        let filtered = block.filter_objects(&labels, &[5, 5, 5], None).unwrap();
        assert_eq!(filtered[(2 * 5 + 2) * 5 + 2], 4);
        assert_eq!(filtered[(2 * 5 + 2) * 5 + 3], 4);
        assert_eq!(filtered.iter().filter(|&&v| v > 0).count(), 2);
    }
}
