import cv2
import numpy as np
from tqdm import tqdm
import datetime
import os
from units import convert_bytes
import argparse

def extract_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()
    return frame

def extract_evenly_spaced_frames(video_path, n):
    """
    Extracts n evenly spaced frames from a video and saves them as images.

    Args:
        video_path (str): Path to the video file.
        n (int): Number of frames to extract.
        output_folder (str): Folder to save the extracted frames.

    Returns:
        list: Captured frames
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames_step = total_frames // (n + 2)
    result = []
    video_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Extracting {n} frames")
    for i in tqdm(range(1, n + 1)):
        frame_number = i * frames_step
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = vidcap.read()

        if success:
            result.append((frame, str(datetime.timedelta(seconds=int(frame_number / fps)))))
            # frame_filename = f"{output_folder}/frame_{i:03d}.jpg"
            # cv2.imwrite(frame_filename, frame)
            # print(f"Saved frame {i+1}/{n} as {frame_filename}")
        else:
            print(f"Error reading frame {i+1}/{n}")

    vidcap.release()
    return result

def create_thumbnail_grid(thumbnails, rows=7, cols=5):
    grid = np.vstack([np.hstack(thumbnails[i:i + cols]) for i in range(0, len(thumbnails), cols)])
    return grid

def merge_images_into_grid(image_list, rows, cols):
    """
    Merges a list of images into a single grid while maintaining aspect ratio.

    Args:
        image_list (list): List of image file paths.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        output_path (str): Path to save the merged image.

    Returns:
        None
    """
    # Load and resize images
    resized_images = []
    common_height = 180

    # Font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .6
    org_1 = (20, 30)
    org_2 = (22, 32)
    font_color_1 = (255, 255, 255)
    font_color_2 = (0, 0, 0)
    thickness = 1


    for img, time in image_list:
        if common_height is None:
            common_height = img.shape[0]
        aspect_ratio = common_height / img.shape[0]
        new_width = int(img.shape[1] * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, common_height))

        cv2.putText(resized_img, time, org_2, font, font_scale, font_color_2, thickness, cv2.LINE_AA)
        cv2.putText(resized_img, time, org_1, font, font_scale, font_color_1, thickness, cv2.LINE_AA)

        resized_images.append(resized_img)

    # Create the grid
    grid = np.vstack([np.hstack(resized_images[i:i + cols]) for i in range(0, len(resized_images), cols)])

    # Save the merged image
    # cv2.imwrite(output_path, grid)
    # print(f"Merged image saved at {output_path}")
    return grid

def add_title(thumbnail, title):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White color
    thickness = 1
    position = (10, 20)  # Position of the title

    cv2.putText(thumbnail, title, position, font, font_scale, font_color, thickness, cv2.LINE_AA)


def add_video_info_section(video_path, img):
    # Load the image
    cap = cv2.VideoCapture(video_path)

    name = os.path.basename(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = str(datetime.timedelta(seconds=total_frames // fps))

    size = convert_bytes(os.path.getsize(video_path))
    bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))

    codec = cap.get(cv2.CAP_PROP_FOURCC)
    pixel = cap.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dimentions = f"{frame_width}x{frame_height}"

    # Create a blank section with the same width as the image
    section_height = 30 * 4  # Adjust as needed
    section = np.zeros((section_height, img.shape[1], 3), dtype=np.uint8)

    # Add text to the section
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)  # White color
    thickness = 1
    text_position = (10, 30)  # Position of the text
    cv2.putText(section, f"Name: {name}", text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(section, f"Duration: {duration} ({size}) {bitrate}kbps", (text_position[0], text_position[1] + 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(section, f"Resolution: {dimentions} {fps} FPS", (text_position[0], text_position[1] + 60), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Combine the image and section
    composite_image = np.vstack((section, img))

    # Save the final image
    return composite_image

def main(argv):
    videos_paths = sorted(argv.videospaths)
    rows = argv.rows
    cols = argv.cols
    overwrite = argv.overwrite

    for i, video_path in enumerate(videos_paths):
        output_path = f"{os.path.splitext(video_path)[0]}_snapshot.jpg"
        print(f"{i + 1}/{len(videos_paths)} {video_path}")
        if not os.path.isfile(output_path) or overwrite:
            thumbnail_list = extract_evenly_spaced_frames(video_path, rows * cols)
            thumbnail_grid = merge_images_into_grid(thumbnail_list, rows, cols)
            thumbnail_with_metadata = add_video_info_section(video_path, thumbnail_grid)

            # prevent incorrect output filename generated by special characters
            is_success, im_buf_arr = cv2.imencode(".jpg", thumbnail_with_metadata)
            im_buf_arr.tofile(output_path)
            # cv2.imwrite(output_path, thumbnail_with_metadata)
        else:
            print("Output file already exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate video thumbnails", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("videospaths", nargs="+", help="Video paths to generate thumbnails")
    parser.add_argument("--rows", "-r", type=int, default=5, help="Number of rows in output image grid")
    parser.add_argument("--cols", "-c", type=int, default=7, help="Number of columns in output image grid")
    parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existent thumbnails")

    argv = parser.parse_args()

    main(argv)