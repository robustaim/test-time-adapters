import cv2
import numpy as np
import time


class SyntheticObjectGenerator:
    """
    A class that generates video frames and bounding box data for
    virtual objects (rectangles) following simple physics laws.
    """
    def __init__(self, width=1280, height=800, min_size=50, max_size=150, max_velocity=15):
        """
        Initialize the generator
        :param width: Frame width
        :param height: Frame height
        :param min_size: Minimum size of the object
        :param max_size: Maximum size of the object
        :param max_velocity: Maximum velocity of the object (pixels/frame)
        """
        self.width = width
        self.height = height
        self.min_size = min_size
        self.max_size = max_size
        self.max_velocity = max_velocity
        self._reset_object()

    def _reset_object(self):
        """Randomly initialize the state (position, size, velocity) of a new object."""
        self.w = np.random.randint(self.min_size, self.max_size)
        self.h = np.random.randint(self.min_size, self.max_size)

        self.x = np.random.randint(0, self.width - self.w)
        self.y = np.random.randint(0, self.height - self.h)

        # Generate random velocity with non-zero sign (-1 or 1)
        vx_sign = np.random.choice([-1, 1])
        vy_sign = np.random.choice([-1, 1])
        self.vx = vx_sign * np.random.randint(5, self.max_velocity)
        self.vy = vy_sign * np.random.randint(5, self.max_velocity)
        print(f"--- New Object Created ---")
        print(f"Position: ({self.x}, {self.y}), Size: ({self.w}, {self.h}), Velocity: ({self.vx}, {self.vy})")

    def _update_state(self):
        """Update the object's position and handle boundary collisions."""
        # Update position
        self.x += self.vx
        self.y += self.vy

        # Handle x-axis boundary collision (Bouncing)
        if self.x <= 0 or self.x + self.w >= self.width:
            self.vx *= -1
            self.x = np.clip(self.x, 0, self.width - self.w)  # Correct to stay within bounds

        # Handle y-axis boundary collision (Bouncing)
        if self.y <= 0 or self.y + self.h >= self.height:
            self.vy *= -1
            self.y = np.clip(self.y, 0, self.height - self.h)  # Correct to stay within bounds

    def _render_frame(self):
        """Generate an image frame and bounding box based on the current state."""
        # Create black background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Current bounding box coordinates (x1, y1, x2, y2 format)
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = int(self.x + self.w), int(self.y + self.h)
        bbox = [x1, y1, x2, y2]

        # Draw white rectangle on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

        return frame, bbox

    def generate_sequence(self, num_frames):
        """
        A generator that produces a video sequence of specified length.
        :param num_frames: Number of frames to generate
        :yields: Tuple of (image frame, bounding box)
        """
        for _ in range(num_frames):
            frame, bbox = self._render_frame()
            yield frame, bbox
            self._update_state()


if __name__ == "__main__":
    # Instantiate the generator
    generator = SyntheticObjectGenerator()

    # Generate and visualize a sequence of 300 frames
    print("\nStarting simulation... Press 'q' to quit.")
    for frame, bbox in generator.generate_sequence(num_frames=300):
        # Add bounding box information as text on the frame
        bbox_text = f"BBox: {bbox}"
        cv2.putText(frame, bbox_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display on screen
        cv2.imshow("Synthetic Object Simulation", frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
