import cv2
import numpy as np
import random


# Simple Particle Class

class Particle:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-2.0, 0.5)
        self.life = random.randint(20, 60)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.05  # slight gravity
        self.life -= 1

    def alive(self):
        return self.life > 0


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Background subtractor = detects movement (difference from background)
    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

    particles = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view for nicer interaction
        h, w = frame.shape[:2]

        # Motion mask
        fg = bg.apply(frame)
        fg = cv2.GaussianBlur(fg, (7, 7), 0)
        _, motion = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # Clean the mask
        motion = cv2.erode(motion, None, iterations=1)
        motion = cv2.dilate(motion, None, iterations=2)

        # Find motion contours
        contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_amount = 0
        motion_points = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 600:  # ignore small noise
                continue

            motion_amount += area
            x, y, cw, ch = cv2.boundingRect(c)

            # Sample a few points inside the motion region to spawn particles
            for _ in range(8):
                px = random.randint(x, x + cw)
                py = random.randint(y, y + ch)
                motion_points.append((px, py))

        # Spawn particles where motion happens
        for (px, py) in motion_points[:120]:
            particles.append(Particle(px, py))

        # Create an "environment layer" (visual effects)
        env = frame.copy()

        # Color shift strength based on motion amount
        strength = min(1.0, motion_amount / 60000.0)

        # Apply a cinematic color tint (cool + warm blend)
        # (No need to mention these exact numbers in report; it's just to look good.)
        cool = np.array([1.0, 1.0, 1.15], dtype=np.float32)   # boost blue slightly
        warm = np.array([1.08, 1.02, 1.0], dtype=np.float32)  # boost red slightly
        tint = (1 - strength) * cool + strength * warm

        env_f = env.astype(np.float32)
        env_f *= tint
        env = np.clip(env_f, 0, 255).astype(np.uint8)

        # "Reveal" effect: show normal camera only where motion is detected
        # Else show tinted environment
        motion_3 = cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR)
        output = np.where(motion_3 > 0, frame, env)

        # Draw and update particles
        for p in particles:
            p.update()
            if 0 <= int(p.x) < w and 0 <= int(p.y) < h:
                cv2.circle(output, (int(p.x), int(p.y)), 2, (255, 255, 255), -1)

        particles = [p for p in particles if p.alive()]

        # UI text
        cv2.putText(output, "Move to generate particles + reveal effect", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Flying Submarine - Webcam Prototype (OpenCV)", output)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
