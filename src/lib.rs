#![recursion_limit = "256"]

pub mod big;
pub mod bioimageio;
pub mod config;
pub mod data;
pub mod fixtures;
pub mod geometry;
pub mod matching;
pub mod model;
pub mod nms;
pub mod package;
pub mod plot;
pub mod rays;
pub mod sample_patches;
#[cfg(feature = "cli")]
pub mod scripts;
pub mod utils;
pub mod weights;

pub use big::{
    _grid_divisible, BigError, BigPolys, BigPolysValue, Block, BlockND, COORD_KEYS,
    NotFullyVisible, OBJECT_KEYS, Polygon, Polyhedron, predict_big,
};
pub use bioimageio::{
    _create_stardist_dependencies, _create_stardist_doc, _get_stardist_metadata,
    _get_weights_and_model_metadata, _import, _predict_tf, BIOIMAGEIO_MISSING_DEPENDENCIES,
    BioimageioAuthor, BioimageioCitation, BioimageioError, BioimageioExport, BioimageioImport,
    BioimageioMode, BioimageioModelConfig, BioimageioPreprocessing, BioimageioThresholds,
    BioimageioWeightsModelMetadata, DEEPIMAGEJ_MACRO, ImportedBioimageio, StardistMetadata,
    export_bioimageio, import_bioimageio,
};
pub use config::{Config2D, Config3D};
pub use data::{
    DataError, GrayU16Image, RgbU8Image, TestImageNuclei2D, TestImageNuclei3D, test_image_he_2d,
    test_image_nuclei_2d, test_image_nuclei_3d,
};
pub use geometry::{
    _cpp_star_dist, _cpp_star_dist3d, _dist_to_coord_old, _ocl_star_dist, _ocl_star_dist3d,
    _polygons_to_label_old, _py_star_dist, _py_star_dist3d, CentroidMode, PolyhedronRenderMode,
    bounding_radius_inner, bounding_radius_inner_gravity, bounding_radius_inner_isotropic,
    bounding_radius_outer, bounding_radius_outer_gravity, bounding_radius_outer_isotropic,
    build_halfspace, calculate_poly_offset_gravity, dist_to_centroid, dist_to_coord,
    dist_to_coord3d, dist_to_volume, export_to_obj_file3d, halfspaces_convex, halfspaces_kernel,
    inside_halfspace, inside_polyhedron, inside_polyhedron_kernel, inside_tetrahedron,
    intersect_bbox, intersect_sphere, intersect_sphere_gravity, intersect_sphere_isotropic,
    overlap_render_polyhedron, overlap_render_polyhedron_kernel, point_in_halfspaces,
    polygons_to_label, polygons_to_label_coord, polyhedron_bbox, polyhedron_centroid,
    polyhedron_polyverts, polyhedron_to_label, polyhedron_volume, qhull_overlap_convex_hulls,
    qhull_overlap_kernel, qhull_volume_halfspace_intersection, ray_angles, relabel_image_stardist,
    relabel_image_stardist3d, render_polyhedron, star_dist, star_dist3d, tetrahedron_volume,
};
pub use matching::{
    _check_label_array, _label_overlap, _safe_divide, _shuffle_labels, DatasetMatchingStats,
    MatchingCriterion, MatchingError, MatchingStats, RelabelSequential, accuracy, f1,
    group_matching_labels, intersection_over_pred, intersection_over_true, intersection_over_union,
    is_array_of_integers, label_are_sequential, label_overlap, matching, matching_dataset,
    matching_dataset_lazy, precision, recall, relabel_sequential,
};
pub use model::{
    _is_multiclass, _parse_classes_arg, _tf_version_at_least, AxesDivByError, AxesError,
    AxesTileOverlapError, ClassesArg, ClassesArgError, ConfigClass, LossError, MaskedPenalty,
    OptimizeThresholdsError, PadMode, PreferredInferenceBackend, ResizerError, StarDist2D,
    StarDist2DInstances, StarDist2DOutputs, StarDist2DPostprocessError,
    StarDist2DPredictInstancesResult, StarDist2DScale, StarDist2DTrainSetup, StarDist3D,
    StarDist3DInstances, StarDist3DPostprocessError, StarDist3DPredictInstancesResult,
    StarDist3DScale, StarDist3DTrainSetup, StarDistBigPrediction, StarDistBigResult,
    StarDistBuildError, StarDistBuildGraph, StarDistBuildLayer, StarDistCheckpointCallback,
    StarDistData2D, StarDistData2DBatch, StarDistData3D, StarDistData3DBatch, StarDistDataBase,
    StarDistDataError, StarDistDirectPrediction, StarDistModelLoadError, StarDistPadAndCropResizer,
    StarDistPredictError, StarDistPredictInstancesBigError, StarDistPredictSetup,
    StarDistPrediction, StarDistPreparedTraining, StarDistSparsePrediction, StarDistThresholds,
    StarDistTrainCallback, StarDistTrainDistLoss, StarDistTrainError,
    StarDistTrainingFinishedAction, ThresholdsError, ThresholdsLoadError, generic_masked_loss, kld,
    masked_loss, masked_loss_iou, masked_loss_mae, masked_loss_mse, masked_metric_iou,
    masked_metric_mae, masked_metric_mse, preferred_inference_backend,
    weighted_categorical_crossentropy,
};
pub use nms::{
    _ind_prob_thresh, _non_maximum_suppression_old, NonMaximumSuppression2D,
    NonMaximumSuppression3D, NonMaximumSuppressionSparse2D, NonMaximumSuppressionSparse3D,
    area_from_path, bbox_intersect, non_maximum_suppression, non_maximum_suppression_3d,
    non_maximum_suppression_3d_inds, non_maximum_suppression_3d_sparse,
    non_maximum_suppression_inds, non_maximum_suppression_sparse, poly_intersection_area,
};
pub use package::{_py_deprecation, STARDIST_VERSION, format_warning};
pub use plot::{
    _draw_polygons, _plot_polygon, _single_color_integer_cmap, PlotDrawCommand, PlotError,
    PlotImage, PlotRange, cmap_from_hls, draw_polygons, match_labels, random_hls,
    random_label_cmap, render_label, render_label_pred,
};
pub use rays::{
    Rays, RaysCartesian, RaysError, RaysExplicit, RaysGoldenSpiral, RaysJson, RaysKwargs, RaysOcto,
    RaysSubDivide, RaysTetra, rays_from_json, reorder_faces,
};
pub use sample_patches::{SamplePatchesError, get_valid_inds, sample_patches};
#[cfg(feature = "cli")]
pub use scripts::{
    PredictScriptArgs, PredictScriptError, PredictScriptImage, PredictScriptLabels,
    PredictScriptOutput,
};
pub use utils::{
    _edt_prob_edt, _edt_prob_scipy, _fill_label_holes, _invert_dict, _is_floatarray,
    _is_power_of_2, _normalize_grid, ArrayDType, ClassAssignment, GridError,
    OptimizeThresholdMeasure, UtilsError, abspath, calculate_extents, edt_prob, export_imagej_rois,
    fill_label_holes, gputools_available, grid_divisible_patch_size, mask_to_categorical,
    optimize_threshold, path_absolute, polyroi_bytearray, sample_points,
};
pub use weights::{KerasWeight, KerasWeights};
