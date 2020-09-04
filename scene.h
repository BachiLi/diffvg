#pragma once

#include "diffvg.h"
#include "aabb.h"
#include <vector>

struct Shape;
struct ShapeGroup;
struct Filter;
struct DFilter;

struct BVHNode {
    int child0, child1; // child1 is negative if it is a leaf
    AABB box;
    float max_radius;
};

struct Scene {
    Scene(int canvas_width,
          int canvas_height,
          const std::vector<const Shape *> &shape_list,
          const std::vector<const ShapeGroup *> &shape_group_list,
          const Filter &filter,
          bool use_gpu,
          int gpu_index);

    ~Scene();

    int canvas_width;
    int canvas_height;

    uint8_t *buffer;

    Shape *shapes;
    Shape *d_shapes;
    ShapeGroup *shape_groups;
    ShapeGroup *d_shape_groups;
    Filter *filter;
    DFilter *d_filter;
    // For accelerating intersection
    AABB *shapes_bbox;
    BVHNode **path_bvhs; // Only for Path
    BVHNode **shape_groups_bvh_nodes; // One BVH for each shape group
    BVHNode *bvh_nodes;

    int num_shapes;
    int num_shape_groups;
    // shape_groups reuse shape, so the total number of shapes
    // doesn't equal to num_shapes
    int num_total_shapes;
    bool use_gpu;
    int gpu_index;

    // For edge sampling
    float *shapes_length;
    float *sample_shapes_cdf;
    float *sample_shapes_pmf;
    int *sample_shape_id;
    int *sample_group_id;
    float **path_length_cdf;
    float **path_length_pmf;
    int **path_point_id_map;

    ShapeGroup get_d_shape_group(int group_id) const;
    Shape get_d_shape(int shape_id) const;
    float get_d_filter_radius() const;
};

struct SceneData {
    int canvas_width;
    int canvas_height;
    Shape *shapes;
    Shape *d_shapes;
    ShapeGroup *shape_groups;
    ShapeGroup *d_shape_groups;
    Filter *filter;
    DFilter *d_filter;
    AABB *shapes_bbox;
    BVHNode **path_bvhs; // Only for Path
    BVHNode **shape_groups_bvh_nodes;
    BVHNode *bvh_nodes;
    int num_shapes;
    int num_shape_groups;
    int num_total_shapes;
    // For edge sampling
    float *shapes_length;
    float *sample_shapes_cdf;
    float *sample_shapes_pmf;
    int *sample_shape_id;
    int *sample_group_id;
    float **path_length_cdf;
    float **path_length_pmf;
    int **path_point_id_map;
};

inline SceneData get_scene_data(const Scene &scene) {
    return SceneData{scene.canvas_width,
                     scene.canvas_height,
                     scene.shapes,
                     scene.d_shapes,
                     scene.shape_groups,
                     scene.d_shape_groups,
                     scene.filter,
                     scene.d_filter,
                     scene.shapes_bbox,
                     scene.path_bvhs,
                     scene.shape_groups_bvh_nodes,
                     scene.bvh_nodes,
                     scene.num_shapes,
                     scene.num_shape_groups,
                     scene.num_total_shapes,
                     scene.shapes_length,
                     scene.sample_shapes_cdf,
                     scene.sample_shapes_pmf,
                     scene.sample_shape_id,
                     scene.sample_group_id,
                     scene.path_length_cdf,
                     scene.path_length_pmf,
                     scene.path_point_id_map};
}
