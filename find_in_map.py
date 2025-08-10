import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RSSMapProcessor:
    def __init__(self, image_path, bs_pixel_coords, bs_real_coords, image_width_meters):
        """
        Initialize the RSS map processor.

        Args:
            image_path: Path to the RSS map image
            bs_pixel_coords: Base station pixel coordinates (px_x, px_y)
            bs_real_coords: Base station real-world coordinates (x, y) in meters
            image_width_meters: Width of the image in meters
        """
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]

        self.bs_pixel_x, self.bs_pixel_y = bs_pixel_coords
        self.bs_real_x, self.bs_real_y = bs_real_coords
        self.image_width_meters = image_width_meters

        # Calculate scale (meters per pixel)
        self.scale = image_width_meters / self.width

        # Calculate the real-world coordinates of the image origin (top-left corner)
        # Since x points left and y points down:
        # - Moving right in pixels means decreasing x in real-world
        # - Moving down in pixels means increasing y in real-world
        self.origin_x = self.bs_real_x + self.bs_pixel_x * self.scale
        self.origin_y = self.bs_real_y - self.bs_pixel_y * self.scale

        print(f"Image dimensions: {self.width}x{self.height} pixels")
        print(f"Scale: {self.scale:.4f} meters/pixel")
        print(f"Origin (top-left) real-world coords: ({self.origin_x:.2f}, {self.origin_y:.2f})")

    def real_to_pixel(self, real_x, real_y):
        """
        Convert real-world coordinates to pixel coordinates.

        Args:
            real_x: Real-world x coordinate (meters)
            real_y: Real-world y coordinate (meters)

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        # Since x points left, larger x values correspond to smaller pixel x
        pixel_x = int((self.origin_x - real_x) / self.scale)
        # Since y points down, larger y values correspond to larger pixel y
        pixel_y = int((real_y - self.origin_y) / self.scale)

        return pixel_x, pixel_y

    def crop_around_user(self, real_x, real_y, crop_size=100):
        """
        Crop the image around a user's location.
    
        Args:
            real_x: User's real-world x coordinate (meters)
            real_y: User's real-world y coordinate (meters)
            crop_size: Size of the crop window in pixels (default 100x100)
    
        Returns:
            Cropped image as numpy array with fixed size (crop_size x crop_size)
        """
        pixel_x, pixel_y = self.real_to_pixel(real_x, real_y)
    
        # Store original coordinates for bounds checking
        original_pixel_x = pixel_x
        original_pixel_y = pixel_y
        
        # Map to nearest point on the map if outside
        pixel_x = np.clip(pixel_x, 0, self.width - 1)
        pixel_y = np.clip(pixel_y, 0, self.height - 1)
    
        # Calculate crop boundaries ensuring we stay within image bounds
        half_size = crop_size // 2
        
        # Adjust center if too close to edges to ensure full crop_size
        center_x = np.clip(pixel_x, half_size, self.width - half_size)
        center_y = np.clip(pixel_y, half_size, self.height - half_size)
        
        # Calculate final crop boundaries
        x1 = center_x - half_size
        y1 = center_y - half_size
        x2 = center_x + half_size
        y2 = center_y + half_size
        
        # Handle edge cases where image is smaller than crop_size
        if self.width < crop_size or self.height < crop_size:
            # Create a crop_size x crop_size array filled with zeros
            cropped = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            
            # Calculate how much of the image we can fit
            img_width = min(self.width, crop_size)
            img_height = min(self.height, crop_size)
            
            # Center the available image in the crop
            offset_x = (crop_size - img_width) // 2
            offset_y = (crop_size - img_height) // 2
            
            cropped[offset_y:offset_y+img_height, offset_x:offset_x+img_width] = \
                self.image_rgb[:img_height, :img_width]
        else:
            # Normal crop
            cropped = self.image_rgb[y1:y2, x1:x2]
            
            # Double-check the size (should always be crop_size x crop_size now)
            if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
                # This should not happen with the adjusted logic, but safety check
                cropped_resized = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                h, w = cropped.shape[:2]
                cropped_resized[:h, :w] = cropped
                cropped = cropped_resized
    
        return cropped

    def visualize_users(self, user_locations, save_path=None):
        """
        Visualize all user locations on the map.

        Args:
            user_locations: List of tuples [(x1, y1), (x2, y2), ...]
            save_path: Optional path to save the visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(self.image_rgb)

        # Mark BS location
        ax.plot(self.bs_pixel_x, self.bs_pixel_y, 'ro', markersize=10, label='Base Station')

        # Mark user locations
        for i, (real_x, real_y) in enumerate(user_locations):
            pixel_x, pixel_y = self.real_to_pixel(real_x, real_y)
            if 0 <= pixel_x < self.width and 0 <= pixel_y < self.height:
                ax.plot(pixel_x, pixel_y, 'b*', markersize=8)
                ax.text(pixel_x + 5, pixel_y - 5, f'U{i+1}', fontsize=8, color='blue')

        ax.set_title('RSS Map with User Locations')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def process_user_dataset(self, user_data_path):
        """
        Process a dataset of user locations.

        Args:
            user_data_path: Path to CSV or text file with user coordinates

        Returns:
            List of cropped images for each user
        """
        # This is a placeholder - implement based on your data format
        # Example: assuming CSV with columns 'x', 'y'
        import pandas as pd

        try:
            df = pd.read_csv(user_data_path)
            user_locations = [(row['x'], row['y']) for _, row in df.iterrows()]
        except:
            # If not CSV, try reading as simple text file with x,y pairs
            user_locations = []
            with open(user_data_path, 'r') as f:
                for line in f:
                    if ',' in line:
                        x, y = map(float, line.strip().split(','))
                        user_locations.append((x, y))

        cropped_images = []
        for i, (x, y) in enumerate(user_locations):
            print(f"\nProcessing user {i+1} at ({x:.2f}, {y:.2f})")
            cropped = self.crop_around_user(x, y)
            if cropped is not None:
                cropped_images.append({
                    'user_id': i+1,
                    'location': (x, y),
                    'image': cropped
                })
                # Save individual crops
                Image.fromarray(cropped).save(f'user_{i+1}_crop.png')
                print(f"Saved crop for user {i+1}")

        return cropped_images


# Example usage
def main():
    # Initialize the processor with your parameters
    processor = RSSMapProcessor(
        image_path='60.jpg',  # Your RSS map image
        bs_pixel_coords=(287, 293),  # BS pixel location
        bs_real_coords=(71.06, 246.29),  # BS real-world location
        image_width_meters=527.5  # Image width in meters
    )

    # Example: Process individual users
    # Test with some example user locations
    example_users = [
        (193, 310),   # User 1
        (-95, 195.0),   # User 2
        (122.188734,51.045463),  # User 3
        # (60.0, 240.0),   # User 4
    ]

    # Visualize all users on the map
    processor.visualize_users(example_users, 'users_on_map.png')

    # Crop around each user
    for i, (x, y) in enumerate(example_users):
        cropped = processor.crop_around_user(x, y, crop_size=30)
        if cropped is not None:
            plt.figure(figsize=(5, 5))
            #plt.imshow(cropped)
            # plt.title(f'User {i+1} at ({x}, {y})')
            #plt.axis('off')
            plt.savefig(f'user_{i+1}_location.png')
            #plt.close()

    # Process from dataset file (when you have the coordinates file)
    # cropped_images = processor.process_user_dataset('user_coordinates.csv')


if __name__ == "__main__":
    main()