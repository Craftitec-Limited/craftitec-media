from os import chmod, mkdir, path, remove
import cv2
import mediapipe as mp
import numpy as np

async def resize_image(
        image, 
        max_width=1024, 
        max_height=768, 
        quality=85):
    """
    Reduces the file size of an image using OpenCV and MediaPipe.

    Args:
        image: The input image as decoded by numpy. See [numpy.frombuffer](https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html) for more information

        max_width: Maximum width of the output image.
        max_height: Maximum height of the output image.
        quality: JPEG compression quality (0-100).

    Returns:
        None
    """

    try:
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Resizes the image
        height, width, _ = img.shape
        scale_factor = min(max_width / width, max_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = cv2.resize(img, (new_width, new_height))

        success, encoded_data = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        return success, encoded_data
    except Exception as exc:
        print("From the resize function")
        raise Exception(str(exc))


async def detect_faces(rgb_image):
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
        min_detection_confidence=0.6,
        model_selection=1
        )
        
        results = face_detection.process(rgb_image)

        face_bounding_boxes = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, c = rgb_image.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                face_bounding_boxes.append((x, y, width, height))

        if len(face_bounding_boxes) < 1:
            raise Exception(
            "No face detected"
            )
        elif len(face_bounding_boxes) > 1:
            raise Exception(
                "More than 1 face detected"
            )


async def store_image(
        image: bytes,
        image_name: str, 
        folder: str,
        masking_folder: str,
        should_detect_faces: bool,
    ) -> str:
    """
    Stores an image onto a folder

    Args:
        image (bytes): The image to be stored, represented in bytes

        image_name (str): The name you want the image to store

        folder (str): The name of the folder you want to store the image to

        masking_folder (str): The name of the folder you want returned

        should_detect_faces (bool): Boolean parameter for if you want to check for faces in the image.
    

    """
    try:
        extension = image_name.split(".")
        if extension[-1] not in ["jpeg","png","jpg","pneg"]:
            raise ValueError(
                f"{extension[-1]} file format is not allowed."
            )
        

        await check_path_exists(folder)

        file_path = f"{folder}/{image_name}"
        
        image_array = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if should_detect_faces:
            await detect_faces(image_rgb)

        success, encoded_image = await resize_image(image_array)
        

        with open(file_path, "wb") as f:
            f.write(encoded_image.tobytes())
            
            chmod(file_path,0o644)
        
        return f"{masking_folder}/{image_name}"
    except Exception as exc:
        raise Exception(str(exc))


async def check_path_exists(media_path: str):
    """
        Check if the path exists

        Args:
            media_path: The path to check
        
        ## Workings:
        - Splits the path by forward slashes
        - Runs a loop that keeps checking if the path exists and creates it if it doesn't

        Returns:
            None
    """
    try:
        split_path = media_path.split("/")
        current_path = split_path[0]

        for i in range(1, len(split_path)):
            current_path += f"/{split_path[i]}"
            if not path.exists(current_path):
                mkdir(current_path)
        
    except Exception as exc:
        raise Exception(str(exc))


async def delete_image(
        image_path: str
):
    try:
        if path.exists(image_path):
            remove(image_path)
            return True
        return False
    except Exception as exc:
        raise Exception(str(exc))