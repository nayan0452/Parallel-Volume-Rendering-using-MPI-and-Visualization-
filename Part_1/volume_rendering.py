from mpi4py import MPI
import numpy as np
from PIL import Image
import time
from scipy.interpolate import interp1d
import sys

#dividing the volume for each processes
def divide_volume(volume_data, num_processes, partition_method, x_min, x_max, y_min, y_max):
    partitions = []

    volume_data = volume_data[y_min:y_max, x_min:x_max, :]  

    if partition_method == 1:
        num_rows = volume_data.shape[0]
        rows_per_process = num_rows // num_processes
        extra_rows = num_rows % num_processes
        start_row = 0
        
        for i in range(num_processes):
            end_row = start_row + rows_per_process + (1 if i < extra_rows else 0)
            partitions.append(volume_data[start_row:end_row, :, :])
            start_row = end_row

    elif partition_method == 2:
        x_divisions, y_divisions = calculate_splits(num_processes)
        total_blocks = x_divisions * y_divisions
        block_size_x = volume_data.shape[0] / x_divisions
        block_size_y = volume_data.shape[1] / y_divisions

        for block_index in range(total_blocks):
            x_block_index = block_index // y_divisions
            y_block_index = block_index % y_divisions

            x_start = int(x_block_index * block_size_x)
            x_end = int((x_block_index + 1) * block_size_x)
            y_start = int(y_block_index * block_size_y)
            y_end = int((y_block_index + 1) * block_size_y)

            x_end = min(x_end, volume_data.shape[0])
            y_end = min(y_end, volume_data.shape[1])

            partitions.append(volume_data[x_start:x_end, y_start:y_end, :])

            if len(partitions) >= num_processes:
                break

    return partitions

#Calculating splits for 3d partitioning
def calculate_splits(num_processes):
    x_divisions = 1
    y_divisions = num_processes

    for i in range(1, int(np.sqrt(num_processes)) + 1):
        if num_processes % i == 0:
            temp_x = i
            temp_y = num_processes // i

            if temp_x < temp_y:
                temp_x, temp_y = temp_y, temp_x

            if abs(temp_x - temp_y) < abs(x_divisions - y_divisions):
                x_divisions, y_divisions = temp_x, temp_y

    return x_divisions, y_divisions

def main():
    #MPI Initialization
    communicator = MPI.COMM_WORLD
    process_rank = communicator.Get_rank()
    num_processes = communicator.Get_size()

    #Taking Arguments
    volume_file = sys.argv[1]
    partition_method = int(sys.argv[2])
    step_interval = float(sys.argv[3])
    x_min = int(sys.argv[4])
    x_max = int(sys.argv[5])
    y_min = int(sys.argv[6])
    y_max = int(sys.argv[7])

    if process_rank == 0:
        load_start_time = time.time()
        volume_data = np.fromfile(volume_file, dtype=np.float32).reshape((1000, 1000, 200), order='f')
        volume_partitions = divide_volume(volume_data, num_processes, partition_method, x_min, x_max, y_min, y_max)
        load_time_elapsed = time.time() - load_start_time
        communicator.scatter(volume_partitions, root=0)
        volume_chunk = volume_partitions[0]
    else:
        volume_chunk = communicator.scatter(None, root=0)

    #Taking opacity transfer function and color transfer function and calculating interpolators
    if process_rank == 0:
        with open('color_tf.txt', 'r') as file:
            color_values = list(map(float, file.read().strip().split(',')))
        scalar_values_color = color_values[0::4]
        color_components = np.array([color_values[i::4] for i in range(1, 4)]).T

        with open('opacity_tf.txt', 'r') as file:
            opacity_values = list(map(float, file.read().strip().split(',')))
        scalar_values_opacity = opacity_values[::2]
        opacity_components = opacity_values[1::2]

        color_r_interp = interp1d(scalar_values_color, color_components[:, 0], kind='linear', bounds_error=False, fill_value=(color_components[0, 0], color_components[-1, 0]))
        color_g_interp = interp1d(scalar_values_color, color_components[:, 1], kind='linear', bounds_error=False, fill_value=(color_components[0, 1], color_components[-1, 1]))
        color_b_interp = interp1d(scalar_values_color, color_components[:, 2], kind='linear', bounds_error=False, fill_value=(color_components[0, 2], color_components[-1, 2]))
        opacity_interp = interp1d(scalar_values_opacity, opacity_components, kind='linear', bounds_error=False, fill_value=(opacity_components[0], opacity_components[-1]))
        interpolators = (color_r_interp, color_g_interp, color_b_interp, opacity_interp)
    else:
        interpolators = None

    interpolators = communicator.bcast(interpolators, root=0)
    color_r_interp, color_g_interp, color_b_interp, opacity_interp = interpolators

    render_start_time = time.time()
    img_height, img_width, depth = volume_chunk.shape
    output_image = np.zeros((img_height, img_width, 3))

    total_rays_cast = 0
    early_terminated_rays = 0

    #Applying Volume Rendering 
    for y_coord in range(img_height):
        for x_coord in range(img_width):
            total_rays_cast += 1
            current_opacity = 0.0
            pixel_color = np.zeros(3)
            depth_coord = 0

            while depth_coord < depth:
                voxel_value = volume_chunk[y_coord, x_coord, int(depth_coord)]
                sampled_color = np.array([color_r_interp(voxel_value), color_g_interp(voxel_value), color_b_interp(voxel_value)])
                sampled_opacity = opacity_interp(voxel_value)

                pixel_color += sampled_color * sampled_opacity * (1 - current_opacity)
                current_opacity += sampled_opacity * (1 - current_opacity)

                if current_opacity >= 0.95: 
                    early_terminated_rays += 1  
                    break

                depth_coord += step_interval

            output_image[y_coord, x_coord, :] = pixel_color
            # print(f"Color for {y_coord},{x_coord},{depth_coord}: {pixel_color}")

    render_time_elapsed = time.time() - render_start_time

    #Gathering Images from each process and Combine then to make final image
    gathered_images = communicator.gather(output_image, root=0)
    gathered_render_times = communicator.gather(render_time_elapsed, root=0)
    gathered_early_terminated_rays = communicator.gather(early_terminated_rays, root=0)
    gathered_total_rays_cast = communicator.gather(total_rays_cast, root=0)

    if process_rank == 0:
        total_rays = sum(gathered_total_rays_cast)
        total_early_terminated = sum(gathered_early_terminated_rays)
        early_termination_fraction = total_early_terminated / total_rays

        print(f"Total rendering time: {max(gathered_render_times):.2f} seconds")
        print(f"Fraction of rays early terminated: {early_termination_fraction:.2%}")

        x_divisions, y_divisions = calculate_splits(num_processes)
        if partition_method == 1:
            final_image = np.vstack(gathered_images)
        else:
            rows = [np.hstack(gathered_images[row_idx * y_divisions:(row_idx + 1) * y_divisions]) for row_idx in range(x_divisions)]
            final_image = np.vstack(rows)

        final_image = (final_image * 255).astype(np.uint8)
        Image.fromarray(final_image).save('Rendered_Image.png')

if __name__ == '__main__':
    main()
