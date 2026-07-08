import warp as wp

@wp.kernel
def box_sdf_kernel(grid_points: wp.array(dtype=wp.vec3),
                   box_centers: wp.array(dtype=wp.vec3),
                   box_sizes: wp.array(dtype=wp.vec3),
                   sdf: wp.array(dtype=float)):
    tid = wp.tid()
    box_id = tid % box_centers.shape[0]
    point_id = tid // box_centers.shape[0]

    p = grid_points[point_id]

    # Get the current box parameters
    box_center = box_centers[box_id]
    half_box_size = box_sizes[box_id] * 0.5

    # Translate point to the box's local space
    p_local = p - box_center

    # Compute SDF for the axis-aligned box in local space
    d = wp.vec3(wp.abs(p_local[0]), wp.abs(p_local[1]), wp.abs(p_local[2])) - half_box_size
    box_sdf = wp.length(wp.max(d, wp.vec3(0., 0., 0.))) + wp.min(wp.max(d), 0.0)
    sdf[tid] = box_sdf
