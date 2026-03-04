import threading
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from ultralytics import YOLO

from sensor_msgs.msg import Image, PointCloud2, CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2

def dominant_color_bgr(
    bgr_img: np.ndarray,
    sat_thresh: int = 40,
    min_ratio: float = 0.03,
    redfam_min_ratio: float = 0.7,
    pink_v_thresh: int = 110,
    pink_s_thresh: int = 190,
) -> Optional[str]:
    if bgr_img is None or bgr_img.size == 0:
        return None

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    valid_mask = s > sat_thresh
    if valid_mask.sum() == 0:
        return None

    h_valid = h[valid_mask]
    s_valid = s[valid_mask]
    v_valid = v[valid_mask]

    color_ranges = {
        "red": [(0, 15)],
        "yellow": [(18, 35)],
        "green": [(40, 85)],
        "blue": [(90, 120)],
        "purple": [(125, 149)],
    }

    def count_range(h_vals: np.ndarray, ranges: list[tuple[int, int]]) -> int:
        total = 0
        for lo, hi in ranges:
            if lo <= hi:
                total += ((h_vals >= lo) & (h_vals <= hi)).sum()
            else:
                total += ((h_vals >= lo) | (h_vals <= hi)).sum()
        return int(total)

    total_pixels = h_valid.size
    color_scores = {
        name: count_range(h_valid, ranges)
        for name, ranges in color_ranges.items()
    }
    best_color, best_count = max(color_scores.items(), key=lambda x: x[1])

    if best_count / total_pixels < min_ratio:
        return None

    if best_color != "red":
        return best_color

    redfam_mask = (
        ((h_valid >= 0) & (h_valid <= 15)) |
        ((h_valid >= 150) & (h_valid <= 179))
    )
    redfam_count = redfam_mask.sum()
    if redfam_count == 0 or redfam_count / total_pixels < redfam_min_ratio:
        return "red"

    s_red = s_valid[redfam_mask].astype(np.float32)
    v_red = v_valid[redfam_mask].astype(np.float32)

    mean_s = float(s_red.mean())
    mean_v = float(v_red.mean())

    if mean_v >= pink_v_thresh and mean_s <= pink_s_thresh:
        return "pink"

    return "red"


class ObjectDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("object_detection")

        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        # Fallback: subscribe to compressed image topic if raw transport is not usable
        self.create_subscription(CompressedImage, '/camera/image/compressed', self.compressed_image_callback, 10)
        self.create_subscription(PointCloud2, 'camera/depth/color/points', self.pointcloud_callback, 10)

        self.declare_parameter("weights_path", r"/home/btnav/repos/Team4-UoM-RSDP/mission_sim/src/rgbd_integration/rgbd_integration/weights.pt")
        self.declare_parameter("img_size", 736)
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("show_window", True)
        self.declare_parameter("loop_hz", 10.0)

        weights_path = self.get_parameter("weights_path").get_parameter_value().string_value
        if not weights_path:
            raise RuntimeError("weights_path parameter is required")

        self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value
        self.conf_threshold = (
            self.get_parameter("conf_threshold").get_parameter_value().double_value
        )
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.show_window = self.get_parameter("show_window").get_parameter_value().bool_value

        loop_hz = self.get_parameter("loop_hz").get_parameter_value().double_value
        if loop_hz <= 0:
            raise RuntimeError("loop_hz must be > 0")

        self.get_logger().info(f"Loading YOLOv8 model from {weights_path}")
        self.model = YOLO(weights_path)
        self.shape_names = self.model.names

        self._shutdown_lock = threading.Lock()
        self._is_shutting_down = False

        self._img_save_count = 0

        self.latest_color_image = None
        self.latest_color_header = None
        self.latest_depth_image = None
        self.latest_depth_intrin = None
        self.latest_pointcloud = None
        self.latest_pointcloud_arr = None
        self.latest_pointcloud_header = None

        # Publishers: color string and 3D position (PointStamped)
        self.color_pub = self.create_publisher(String, "/object_detection/color", 10)
        self.position_pub = self.create_publisher(PointStamped, "/object_detection/position", 10)
        self.timer = self.create_timer(1.0 / loop_hz, self.process_frame)

    def process_frame(self) -> None:
        if self._is_shutting_down:
            return

        color_image = self.latest_color_image
        depth_image = self.latest_depth_image
        depth_intrin = self.latest_depth_intrin
        pc_arr = self.latest_pointcloud_arr

        # If no color image, show a placeholder window and return
        if color_image is None:
            if self.show_window:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Waiting for image...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("ROS Color+Shape", placeholder)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.request_shutdown()
            return

        # Log image shape occasionally for debugging
        if hasattr(color_image, "shape"):
            h_img, w_img = color_image.shape[:2]
            self.get_logger().debug(f"Got color image {w_img}x{h_img}")

        results = self.model(
            source=color_image,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            stream=False,
        )
        res = results[0]
        boxes = res.boxes

        if boxes is None or boxes.xyxy.shape[0] == 0:
            if self.show_window:
                cv2.imshow("ROS Color+Shape", color_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.request_shutdown()
            return

        h_img, w_img = color_image.shape[:2]

        for i in range(boxes.xyxy.shape[0]):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            shape_name = self.shape_names.get(cls_id, str(cls_id))

            x1 = max(0, min(x1, w_img - 1))
            x2 = max(0, min(x2, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            y2 = max(0, min(y2, h_img - 1))
            if x2 <= x1 or y2 <= y1:
                roi = color_image
            else:
                roi = color_image[y1:y2, x1:x2]

            pred_color = dominant_color_bgr(roi) or "unknown"

            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            X = Y = Z = float("nan")
            depth = float("nan")
            # Prefer organized pointcloud array if available (x,y,z per pixel)
            if pc_arr is not None:
                try:
                    pt = pc_arr[y_center, x_center]
                    X, Y, Z = float(pt[0]), float(pt[1]), float(pt[2])
                    depth = Z
                except Exception:
                    pass
            elif depth_image is not None and depth_intrin is not None:
                depth = float(depth_image[y_center, x_center])
                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [x_center, y_center], depth)

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{pred_color} {shape_name} {conf:.2f}"
            cv2.putText(
                color_image,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            depth_text = f"{depth:.2f}m"
            cv2.putText(
                color_image,
                depth_text,
                (x1, y2 + 20 if y2 + 20 < h_img else y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            self.get_logger().info(
                f"[DETECT] {pred_color} {shape_name} | "
                f"conf={conf:.2f} | depth={depth:.3f}m | "
                f"X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}"
            )

            # Publish color
            try:
                color_msg = String()
                color_msg.data = pred_color
                self.color_pub.publish(color_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish color: {e}")

            # Publish position as PointStamped (use image header if available)
            try:
                pst = PointStamped()
                hdr = pst.header
                if self.latest_color_header is not None:
                    hdr.stamp = self.latest_color_header.stamp
                    hdr.frame_id = self.latest_color_header.frame_id
                else:
                    hdr.stamp = self.get_clock().now().to_msg()
                    hdr.frame_id = "camera_link"
                pst.point.x = float(X)
                pst.point.y = float(Y)
                pst.point.z = float(Z)
                self.position_pub.publish(pst)
            except Exception as e:
                self.get_logger().error(f"Failed to publish position: {e}")

        if self.show_window:
            cv2.imshow("ROS Color+Shape", color_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.request_shutdown()

    def request_shutdown(self) -> None:
        with self._shutdown_lock:
            if self._is_shutting_down:
                return
            self._is_shutting_down = True
        self.get_logger().info("Shutdown requested")
        rclpy.shutdown()

    def destroy_node(self) -> bool:
        with self._shutdown_lock:
            self._is_shutting_down = True
        if self.show_window:
            cv2.destroyAllWindows()
        return super().destroy_node()
    def image_callback(self, msg: Image) -> None:
        try:
            enc = getattr(msg, 'encoding', '')
            self.get_logger().debug(f"image_callback: encoding={enc} h={msg.height} w={msg.width} step={msg.step}")

            candidates = []
            # Try common desired encodings
            for desired in ('bgr8', 'rgb8', 'passthrough'):
                try:
                    arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding=desired)
                    # If rgb, convert to bgr for OpenCV display
                    if desired == 'rgb8' and arr is not None:
                        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    candidates.append((desired, arr))
                except Exception as _:
                    candidates.append((desired, None))

            # Heuristic: choose first candidate with 3 channels and reasonable dynamic range
            chosen = None
            diag_lines = []
            for name, arr in candidates:
                if arr is None:
                    diag_lines.append(f"{name}: conversion failed")
                    continue
                if not (hasattr(arr, 'ndim') and arr.ndim == 3 and arr.shape[2] == 3):
                    diag_lines.append(f"{name}: wrong shape {getattr(arr,'shape',None)}")
                    continue
                lo, hi = float(arr.min()), float(arr.max())
                diag_lines.append(f"{name}: shape={arr.shape} dtype={arr.dtype} min={lo} max={hi}")
                if hi - lo > 5.0:
                    chosen = arr
                    diag_lines.append(f"{name}: selected")
                    break

            # If none chosen, try raw buffer reshape fallback
            if chosen is None:
                try:
                    raw = np.frombuffer(msg.data, dtype=np.uint8)
                    step = int(msg.step)
                    h = int(msg.height)
                    w = int(msg.width)
                    if raw.size >= h * step:
                        raw2 = raw[: h * step].reshape((h, step))
                        # take first 3*w columns (assuming interleaved RGB)
                        img = raw2[:, : (3 * w)].reshape((h, w, 3)).copy()
                        # assume rgb8 from topic
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        lo, hi = float(img.min()), float(img.max())
                        diag_lines.append(f"raw_fallback: shape={img.shape} min={lo} max={hi}")
                        if hi - lo > 5.0:
                            chosen = img
                except Exception as e:
                    diag_lines.append(f"raw_fallback failed: {e}")

            # Write diagnostics
            try:
                with open('/tmp/object_detection_debug.txt', 'w') as f:
                    f.write('\n'.join(diag_lines))
            except Exception:
                pass

            if chosen is None:
                self.get_logger().warning("No valid image conversion found; saving diagnostics")
                self.latest_color_image = None
                return

            cv_image = chosen
            self.latest_color_image = cv_image
            self.latest_color_header = msg.header
            self._img_save_count += 1
            if self._img_save_count % 10 == 1:
                try:
                    cv2.imwrite('/tmp/object_detection_last.jpg', cv_image)
                    np.save('/tmp/object_detection_last.npy', cv_image)
                except Exception as e:
                    self.get_logger().warning(f"Failed to write debug image: {e}")
            # Optionally, store header or other info
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def compressed_image_callback(self, msg: CompressedImage) -> None:
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warning("Failed to decode CompressedImage")
                return
            self.latest_color_image = img
            # CompressedImage doesn't have header in ROS1; in ROS2 it does
            try:
                self.latest_color_header = msg.header
            except Exception:
                self.latest_color_header = None
            # Save occasional debug
            self._img_save_count += 1
            if self._img_save_count % 10 == 1:
                try:
                    cv2.imwrite('/tmp/object_detection_last_compressed.jpg', img)
                except Exception:
                    pass
        except Exception as e:
            self.get_logger().error(f"Failed to handle CompressedImage: {e}")

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        # Convert organized PointCloud2 to numpy array (height x width x 3) if possible
        try:
            gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False)
            pts_list = []
            for p in gen:
                try:
                    x, y, z = p
                    pts_list.append([float(x), float(y), float(z)])
                except Exception:
                    continue
            if len(pts_list) == 0:
                return
            arr = np.array(pts_list, dtype=np.float32)
            if msg.height > 1 and msg.width > 0:
                try:
                    arr = arr.reshape((msg.height, msg.width, 3))
                    self.latest_pointcloud_arr = arr
                    self.latest_pointcloud_header = msg.header
                    # Also store a depth image (Z channel)
                    self.latest_depth_image = arr[:, :, 2]
                except Exception as e:
                    self.get_logger().warning(f"PointCloud reshape failed: {e}")
                    self.latest_pointcloud_arr = None
            else:
                # Unorganized cloud: store as flat array (not usable as image)
                self.latest_pointcloud_arr = None
            self.latest_pointcloud = msg
        except Exception as e:
            self.get_logger().error(f"Failed to convert PointCloud2: {e}")


def main() -> None:
    rclpy.init()
    try:
        node = ObjectDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
