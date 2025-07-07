import torch
import numpy as np
from PIL import Image
import cv2
import rembg
import mediapipe as mp
from diffusers import StableDiffusionInpaintPipeline
from transformers import pipeline
import torchvision.transforms as transforms


class VTONPreprocessor:
    def __init__(self, device="cpu"):
        self.device = device
        self.bg_remover = rembg.create_new_session("u2net")
        self.pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
        self.segmenter = pipeline(
            "image-segmentation", model="mattmdjaga/segformer_b2_clothes"
        )

    def process_person(self, person_img):
        # Convert to numpy array
        person_np = np.array(person_img)

        # Remove background
        person_nobg = rembg.remove(person_np, session=self.bg_remover)
        person_nobg = Image.fromarray(person_nobg)

        # Extract pose keypoints
        rgb_image = cv2.cvtColor(person_np, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.visibility])

        # Resize to standard size
        person_resized = person_nobg.resize((768, 1024))

        return {
            "image": person_resized,
            "keypoints": np.array(keypoints),
            "original_size": person_img.size,
        }

    def process_garment(self, garment_img):
        # Segment garment
        segments = self.segmenter(garment_img)

        # Find largest clothing segment
        max_area = 0
        best_mask = None

        for segment in segments:
            mask = np.array(segment["mask"])
            area = np.sum(mask)
            if area > max_area:
                max_area = area
                best_mask = mask

        if best_mask is None:
            # Fallback: use whole image
            best_mask = (
                np.ones((garment_img.size[1], garment_img.size[0]), dtype=np.uint8)
                * 255
            )

        # Apply mask to garment
        garment_np = np.array(garment_img)
        garment_masked = garment_np.copy()
        garment_masked[best_mask == 0] = [255, 255, 255]  # White background

        garment_final = Image.fromarray(garment_masked).resize((768, 1024))
        mask_final = Image.fromarray(best_mask).resize((768, 1024))

        return {"image": garment_final, "mask": mask_final}

    def warp_garment_to_person(self, garment_data, person_data):
        # Simple warping based on person dimensions
        # In a real implementation, this would use TPS or advanced warping

        person_keypoints = person_data["keypoints"]
        garment_img = garment_data["image"]

        if len(person_keypoints) > 0:
            # Calculate person dimensions from keypoints
            shoulder_width = (
                abs(person_keypoints[11][0] - person_keypoints[12][0])
                if len(person_keypoints) > 12
                else 0.3
            )
            torso_height = (
                abs(person_keypoints[11][1] - person_keypoints[23][1])
                if len(person_keypoints) > 23
                else 0.4
            )

            # Scale garment accordingly
            new_width = int(768 * shoulder_width * 1.2)
            new_height = int(1024 * torso_height * 1.2)

            # Ensure minimum size
            new_width = max(new_width, 200)
            new_height = max(new_height, 300)

            warped_garment = garment_img.resize((new_width, new_height))

            # Center the garment on a canvas
            canvas = Image.new("RGBA", (768, 1024), (255, 255, 255, 0))
            x_offset = (768 - new_width) // 2
            y_offset = int(1024 * 0.2)  # Position in upper body area

            canvas.paste(warped_garment, (x_offset, y_offset))
            return canvas

        return garment_img


class DualMeVTON:
    def __init__(self, device="cpu"):
        self.device = device
        print("🔧 Loading DualMe VTON generative model...")

        try:
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            self.pipeline = None

    def generate(self, person_data, warped_garment):
        if self.pipeline is None:
            # Fallback: simple composition
            person_img = person_data["image"].convert("RGBA")
            garment_img = warped_garment.convert("RGBA")

            # Simple alpha blending
            result = Image.alpha_composite(person_img, garment_img)
            return result.convert("RGB")

        try:
            # Prepare inputs for inpainting
            person_img = person_data["image"].convert("RGB")
            garment_img = warped_garment.convert("RGB")

            # Create mask for clothing area (simplified)
            mask = Image.new(
                "L", person_img.size, 255
            )  # White mask for inpainting area

            # Create inpainting prompt
            prompt = "person wearing clothing, high quality, realistic, detailed fabric texture"
            negative_prompt = "blurry, distorted, low quality, artifacts"

            # Generate result
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=person_img,
                mask_image=mask,
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]

            return result

        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            # Fallback to simple composition
            person_img = person_data["image"].convert("RGBA")
            garment_img = warped_garment.convert("RGBA")
            result = Image.alpha_composite(person_img, garment_img)
            return result.convert("RGB")
