import numpy as np
import sys
import matplotlib.pyplot as plt
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD  # Initialize MPI communicator
    n_procs = comm.Get_size()  # Number of processes
    rank = comm.Get_rank()  # Rank of the current process

    start_time_total = MPI.Wtime()  # Record total start time

    send_duration = 0.0  # Duration for sending data
    recv_duration = 0.0  # Duration for receiving data
    gather_duration = 0.0  # Duration for gathering data

    # Input parameters from command-line arguments
    dataset_filename = sys.argv[1]  # Path to the dataset file
    x_divs, y_divs, z_divs = map(int, sys.argv[2:5])  # Number of divisions in x, y, and z dimensions
    step = float(sys.argv[5])  # Step size for ray casting
    opacity_tf_file = sys.argv[6]  # Path to opacity transfer function file
    color_tf_file = sys.argv[7]  # Path to color transfer function file

    # Predefined dataset dimensions for validation
    dataset_dimensions = {
        "1000x1000x200": (1000, 1000, 200),
        "2000x2000x400": (2000, 2000, 400)
    }

    # Determine dataset dimensions based on filename
    try:
        data_dims = next(dims for key, dims in dataset_dimensions.items() if key in dataset_filename)
    except StopIteration:
        raise ValueError("Unknown dataset dimensions in filename.")

    if rank == 0:
        # Load the dataset and split it into subdomains across the processes
        load_start = MPI.Wtime()
        data_volume = np.fromfile(dataset_filename, dtype=np.float32).reshape(data_dims, order='F')

        # Split data into subdomains based on x, y, z divisions
        x_splits = np.array_split(data_volume, x_divs, axis=0)
        data_parts = [np.array_split(part, y_divs, axis=1) for part in x_splits]

        # Calculate subdomain shape for each rank
        x_step = data_dims[0] // x_divs
        y_step = data_dims[1] // y_divs
        z_step = data_dims[2] // z_divs

        # Start non-blocking sends to distribute subdomains to other processes
        send_requests = []
        for ix in range(x_divs):
            for iy in range(y_divs):
                for iz in range(z_divs):
                    target_rank = ix * y_divs * z_divs + iy * z_divs + iz
                    sub_vol = data_parts[ix][iy][:, :, iz::z_divs]
                    sub_vol = np.ascontiguousarray(sub_vol)  # Ensure contiguity of data
                    if target_rank == 0:
                        local_volume = sub_vol
                    else:
                        req = comm.Isend(sub_vol, dest=target_rank, tag=target_rank)
                        send_requests.append(req)
        MPI.Request.Waitall(send_requests)  # Wait for all non-blocking sends to complete
        send_duration = MPI.Wtime() - load_start
        print(f"Data loading and distribution took {send_duration:.4f} seconds")
    else:
        # Receive data for each process with matching buffer shape
        x_step = data_dims[0] // x_divs
        y_step = data_dims[1] // y_divs
        z_step = data_dims[2] // z_divs
        recv_start = MPI.Wtime()
        local_volume = np.empty((x_step, y_step, z_step), dtype=np.float32)
        comm.Recv(local_volume, source=0, tag=rank)  # Receive the subdomain data
        recv_duration = MPI.Wtime() - recv_start

    # Function to load transfer functions (opacity or color)
    def load_tf(filepath, is_color=True):
        with open(filepath, 'r') as file:
            values = [float(val) for line in file for val in line.strip().replace(',', '').split()]
        # Parse opacity or color transfer function based on is_color flag
        return [(values[i], tuple(values[i + 1:i + 4])) for i in range(0, len(values), 4)] if is_color else [(values[i], values[i + 1]) for i in range(0, len(values), 2)]

    # Load opacity and color transfer functions
    opacity_tf = load_tf(opacity_tf_file, is_color=False)
    color_tf = load_tf(color_tf_file, is_color=True)

    # Ray casting procedure for each subdomain to generate the image section
    ray_start = MPI.Wtime()
    h, w, d = local_volume.shape
    image_section = np.zeros((h, w, 3))  # Initialize the image section for this subdomain

    # Create a meshgrid of pixel indices for the 2D subdomain
    x_idx, y_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Reshape indices to iterate over all pixels in the subdomain at once
    x_idx = x_idx.flatten()
    y_idx = y_idx.flatten()

    # Iterate over each pixel in the subdomain using flattened indices
    for idx in range(len(x_idx)):
        x, y = x_idx[idx], y_idx[idx]
    
        color_accum = np.zeros(3)  # Accumulated color for the pixel
        opacity_accum = 0  # Accumulated opacity
        depth_pos = 0.0  # Initial depth position for ray casting

        # Vectorize the depth loop by handling depth positions in a range
        while depth_pos < d:
            d_start = int(depth_pos)
            d_end = min(d_start + 1, d - 1)
            interp_factor = depth_pos - d_start
            voxel_val = (1 - interp_factor) * local_volume[x, y, d_start] + interp_factor * local_volume[x, y, d_end]

            # Vectorized Color Interpolation based on the transfer function
            color = np.zeros(3)
            for i in range(len(color_tf) - 1):
                v0, c0 = color_tf[i]
                v1, c1 = color_tf[i + 1]
                if v0 <= voxel_val <= v1:
                    interp_factor = (voxel_val - v0) / (v1 - v0)
                    color = np.array([c0[j] + interp_factor * (c1[j] - c0[j]) for j in range(3)])
                    break

            # Vectorized Opacity Interpolation based on the transfer function
            opacity = 0
            for i in range(len(opacity_tf) - 1):
                v0, o0 = opacity_tf[i]
                v1, o1 = opacity_tf[i + 1]
                if v0 <= voxel_val <= v1:
                    if v1 != v0:
                        opacity = o0 + ((o1 - o0) * (voxel_val - v0) / (v1 - v0))
                    else:
                        opacity = o0
                    break

            # Update accumulated color and opacity
            color_accum += (1 - opacity_accum) * color * opacity
            opacity_accum += (1 - opacity_accum) * opacity

            # Early exit if opacity reaches a threshold (for optimization)
            if opacity_accum >= 0.98:
                break

            depth_pos += step

        # Store the accumulated color in the image section for this pixel
        image_section[x, y, :] = color_accum  # Store the color for the current pixel

    # Calculate the ray casting time
    ray_duration = MPI.Wtime() - ray_start  # Total time for ray casting

    # Gather image sections from all processes to the root process
    gather_start = MPI.Wtime()
    gathered_sections = comm.gather(image_section, root=0)
    gather_duration = MPI.Wtime() - gather_start

    if rank == 0:
        print("Assembling the final image...")

        # Get the shape of one section
        section_h, section_w, _ = gathered_sections[0].shape

        # Create an empty array to hold the final image
        final_image = np.zeros((section_h * x_divs, section_w * y_divs, 3))

        # Generate meshgrid of indices for ix and iy (to place sections in the final image)
        ix_vals, iy_vals = np.meshgrid(np.arange(x_divs), np.arange(y_divs), indexing='ij')
        ix_vals = ix_vals.flatten()
        iy_vals = iy_vals.flatten()

        # Loop over ix and iy using a single loop
        for idx in range(len(ix_vals)):
            ix, iy = ix_vals[idx], iy_vals[idx]

            final_color = np.zeros((section_h, section_w, 3))
            final_opacity = np.zeros((section_h, section_w))

            # Loop over iz (depth slices) and accumulate the image sections
            for iz in range(z_divs):
                gather_idx = ix * y_divs * z_divs + iy * z_divs + iz
                sub_img = gathered_sections[gather_idx]

                alpha = 1 - final_opacity
                final_color += sub_img * alpha[:, :, None]
                final_opacity += alpha

            # Place the final color for this block into the appropriate region of the full image
            final_image[ix * section_h:(ix + 1) * section_h, iy * section_w:(iy + 1) * section_w, :] = final_color

        # Optionally save or display the image after assembling
        output_file = f"{x_divs}_{y_divs}_{z_divs}.png"
        plt.imsave(output_file, final_image)
        print(f"Final image saved : {output_file}")

    # Communication statistics across all processes
    max_ray_duration = comm.reduce(ray_duration, op=MPI.MAX, root=0)
    max_send_duration = comm.reduce(send_duration, op=MPI.MAX, root=0)
    max_recv_duration = comm.reduce(recv_duration, op=MPI.MAX, root=0)
    max_gather_duration = comm.reduce(gather_duration, op=MPI.MAX, root=0)

    if rank == 0:
        # Only root process prints the summary of timings
        print(f"Data loading and distribution took {send_duration:.4f} seconds")
        print(f"Final image saved : 2_2_2.png")
        print(f"Computation Time (Max Ray casting time) (in seconds) : {max_ray_duration:.4f} seconds")
        print(f"Total communication time (in seconds) : {max_send_duration + max_recv_duration + max_gather_duration:.4f} seconds")
        total_duration = max_ray_duration + max_send_duration + max_recv_duration + max_gather_duration
        print(f"Total execution time (in seconds) : {total_duration:.4f} seconds")

if __name__ == "__main__":
    main()