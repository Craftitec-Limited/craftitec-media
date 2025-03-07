from os import chmod, mkdir, path, remove
import cv2
import mediapipe as mp
import numpy as np

async def resize_image(
        img, 
        max_width=1024, 
        max_height=768, 
        quality=85):
    """
    Resizes an image while maintaining aspect ratio.
    """
    try:
        height, width, _ = img.shape
        scale_factor = min(max_width / width, max_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = cv2.resize(img, (new_width, new_height))

        success, encoded_data = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return success, encoded_data
    except Exception as exc:
        raise Exception(f"Error in resize_image: {exc}")



async def detect_faces(rgb_image):
    """
    Detects faces and returns bounding boxes instead of just raising exceptions.
    """
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=0.6,
            model_selection=1
        )
        
        results = face_detection.process(rgb_image)
        face_bounding_boxes = []
        
        if results.detections:
            h, w, _ = rgb_image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                face_bounding_boxes.append((x, y, width, height))

        return face_bounding_boxes
    except Exception as exc:
        raise Exception(f"Error in detect_faces: {exc}")

async def store_image(
        image: bytes,
        image_name: str, 
        folder: str,
        masking_folder: str,
        should_detect_faces: bool,
    ) -> str:
    """
    Stores an image onto a folder and optionally detects faces.
    """
    try:
        extension = image_name.split(".")[-1].lower()
        allowed_formats = ["jpeg", "png", "jpg"]
        
        if extension not in allowed_formats:
            raise ValueError(f"Invalid file format: {extension}")

        await check_path_exists(folder)
        file_path = f"{folder}/{image_name}"
        
        image_array = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image. Ensure it is a valid format.")

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if should_detect_faces:
            face_boxes = await detect_faces(image_rgb)
            if len(face_boxes) < 1:
                raise ValueError("No face detected")
            elif len(face_boxes) > 1:
                raise ValueError("More than one face detected")

        success, encoded_image = await resize_image(img)

        if not success:
            raise ValueError("Image encoding failed")

        with open(file_path, "wb") as f:
            f.write(encoded_image.tobytes())
            chmod(file_path, 0o644)
        
        return f"{masking_folder}/{image_name}"
    except Exception as exc:
        raise Exception(f"Error in store_image: {exc}")


async def check_path_exists(media_path: str):
    """
    Ensures that all directories in the given path exist.
    """
    try:
        if not path.exists(media_path):
            mkdir(media_path)
    except Exception as exc:
        raise Exception(f"Error in check_path_exists: {exc}")


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